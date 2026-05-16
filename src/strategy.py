"""
Phase C3: Federated Learning Strategy
AdaptiveAggregationStrategy implements FedAvg/FedSGD switching with static τ.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar, FitRes, EvaluateRes, Parameters, GetParametersIns
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.aggregators import aggregate_adaptive, compute_divergence


class AdaptiveAggregationStrategy(Strategy):
    """
    Federated Learning strategy with adaptive aggregation.
    Switches between FedAvg and FedSGD based on client divergence threshold τ.
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = config.NUM_CLIENTS_BASELINE,
        min_evaluate_clients: int = config.NUM_CLIENTS_BASELINE,
        min_available_clients: int = config.NUM_CLIENTS_BASELINE,
        evaluate_fn=None,
        tau: float = config.TAU_STATIC,
        learning_rate: float = config.LEARNING_RATE,
    ):
        """
        Args:
            fraction_fit: Fraction of clients sampled for training
            fraction_evaluate: Fraction of clients sampled for evaluation
            min_fit_clients: Minimum number of clients for training round
            min_evaluate_clients: Minimum number of clients for evaluation round
            min_available_clients: Minimum number of available clients
            evaluate_fn: Optional server-side evaluation function
            tau: Divergence threshold for FedAvg/FedSGD switching
            learning_rate: Server-side learning rate for FedSGD
        """
        super().__init__()
        
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.tau = tau
        self.learning_rate = learning_rate
        
        # Tracking metrics
        self.round = 0
        self.divergence_history = []
        self.algorithm_history = []  # Track which algorithm was used per round
        self.loss_history = []
        self.val_acc_history = []
        self.final_parameters = None  # Track final aggregated parameters
        
        logger.info(
            f"AdaptiveAggregationStrategy initialized: "
            f"tau={tau}, lr={learning_rate}, "
            f"min_fit_clients={min_fit_clients}"
        )
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """
        Initialize global model parameters.
        Requests from first available client if not overridden.
        """
        sample = client_manager.sample(1)
        if sample:
            client = sample[0]
            ins = GetParametersIns(config={})
            res = client.get_parameters(ins, timeout=None, group_id=0)
            logger.info("Global parameters initialized from first client")
            return res.parameters
        
        logger.warning("No clients available for initialization")
        return None
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Configure the fit round: sample clients and set their config.
        """
        sample_size = max(
            self.min_fit_clients,
            int(client_manager.num_available() * self.fraction_fit),
        )
        sample = client_manager.sample(sample_size)
        
        fit_config = {
            "server_round": server_round,
            "num_epochs": config.NUM_EPOCHS_PER_ROUND,
            "batch_size": config.BATCH_SIZE,
        }
        
        fit_ins = fl.common.FitIns(parameters, fit_config)
        
        return [(client, fit_ins) for client in sample]
    
    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using adaptive FedAvg/FedSGD.
        """
        self.round = server_round
        
        if not results:
            logger.warning("No client results to aggregate")
            return None, {}
        
        # Extract weights and sample counts
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        client_weights = [w for w, _ in weights_results]
        sample_counts = [n for _, n in weights_results]
        
        # Get current global parameters for divergence calculation
        # (We'll use last aggregated as reference)
        if not hasattr(self, 'global_params'):
            # Initialize with first client params
            self.global_params = client_weights[0]
        
        # Compute divergence
        divergence = compute_divergence(client_weights, self.global_params)
        self.divergence_history.append(divergence)
        
        # Aggregate with adaptive strategy
        aggregated, algorithm_used, div = aggregate_adaptive(
            client_weights,
            self.global_params,
            tau=self.tau,
            learning_rate=self.learning_rate,
            client_sample_counts=sample_counts,
        )
        
        self.global_params = aggregated
        self.algorithm_history.append(algorithm_used)
        
        # Log metrics
        total_loss = sum(
            fit_res.metrics.get("loss", 0.0) * fit_res.num_examples
            for _, fit_res in results
        )
        avg_loss = total_loss / sum(sample_counts) if sample_counts else 0
        self.loss_history.append(avg_loss)
        
        # Aggregate privacy metrics if available
        privacy_metrics = {}
        epsilon_values = [
            fit_res.metrics.get("epsilon", float('inf'))
            for _, fit_res in results
        ]
        if any(e != float('inf') for e in epsilon_values):
            privacy_metrics["epsilon"] = min(epsilon_values)
        
        metrics = {
            "loss": avg_loss,
            "divergence": divergence,
            "algorithm": algorithm_used,
            "tau": self.tau,
            "num_clients": len(results),
            **privacy_metrics,
        }
        
        logger.info(
            f"Round {server_round}: loss={avg_loss:.4f}, div={divergence:.6f}, "
            f"algorithm={algorithm_used}, epsilon={metrics.get('epsilon', 'N/A')}"
        )
        
        # Store final parameters after each round
        self.final_parameters = fl.common.ndarrays_to_parameters(aggregated)
        
        return self.final_parameters, metrics
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """
        Configure evaluation round.
        """
        sample_size = max(
            self.min_evaluate_clients,
            int(client_manager.num_available() * self.fraction_evaluate),
        )
        sample = client_manager.sample(sample_size)
        
        eval_ins = fl.common.EvaluateIns(parameters, {})
        
        return [(client, eval_ins) for client in sample]
    
    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results.
        """
        if not results:
            logger.warning("No evaluation results to aggregate")
            return None, {}
        
        loss_values = [eva_res.loss * eva_res.num_examples for _, eva_res in results]
        examples = [eva_res.num_examples for _, eva_res in results]
        
        aggregated_loss = sum(loss_values) / sum(examples) if examples else 0
        self.val_acc_history.append(
            sum(
                eva_res.metrics.get("accuracy", 0.0) * eva_res.num_examples
                for _, eva_res in results
            ) / sum(examples) if examples else 0
        )
        
        metrics = {
            "val_loss": aggregated_loss,
            "val_accuracy": self.val_acc_history[-1],
        }
        
        logger.info(
            f"Round {server_round} evaluation: "
            f"loss={aggregated_loss:.4f}, accuracy={self.val_acc_history[-1]:.4f}"
        )
        
        return aggregated_loss, metrics
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Optional server-side evaluation.
        """
        if self.evaluate_fn is None:
            return None
        
        weights = fl.common.parameters_to_ndarrays(parameters)
        eval_result = self.evaluate_fn(server_round, weights, {})
        
        return eval_result
    
    def get_metrics(self) -> Dict:
        """Return aggregated metrics history."""
        return {
            "divergence_history": self.divergence_history,
            "algorithm_history": self.algorithm_history,
            "loss_history": self.loss_history,
            "val_acc_history": self.val_acc_history,
        }


if __name__ == "__main__":
    # Test strategy creation
    strategy = AdaptiveAggregationStrategy(
        tau=0.10,
        learning_rate=0.01,
    )
    print(f"Strategy created: {strategy}")
    print(f"Metrics: {strategy.get_metrics()}")

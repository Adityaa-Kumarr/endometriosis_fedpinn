import flwr as fl
import sys

def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    # In a real scenario, the server might have a global validation set.
    # For now, we rely on client-side evaluation.
    return None

def start_server(num_rounds=5, fraction_fit=1.0, fraction_evaluate=1.0):
    """
    Start the federated learning server using strategy such as FedAvg or FedProx.
    """
    # Create strategy
    strategy = fl.server.strategy.FedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
        proximal_mu=0.1,  # FedProx hyperparameter to handle statistical heterogeneity
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"Starting FedProx Server for {rounds} rounds...")
    start_server(num_rounds=rounds)

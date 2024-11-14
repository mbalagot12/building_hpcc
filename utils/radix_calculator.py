import pandas as pd
from tabulate import tabulate


def calculate_radix(spine_ports, leaf_uplink_ports):
    """
    Calculates the theoretical maximum radix of a leaf-spine network.

    Args:
      spine_ports: The total number of available ports on the spine switch.
      leaf_uplink_ports: The number of ports used per leaf switch for uplinks to the spine.

    Returns:
      The theoretical maximum radix (number of leaf switches).
    """
    return spine_ports // leaf_uplink_ports


def calculate_nodes_per_leaf(leaf_bandwidth, node_bandwidth, uplink_bandwidth):
    """
    Calculates the number of nodes each leaf can support.

    Args:
      leaf_bandwidth: Total throughput of the leaf switch (in Gbps).
      node_bandwidth: Bandwidth of each compute node's NIC (in Gbps).
      uplink_bandwidth: Total uplink bandwidth used by the leaf (in Gbps).

    Returns:
      The number of nodes each leaf can support.
    """
    available_bandwidth = leaf_bandwidth - uplink_bandwidth
    return int(available_bandwidth // node_bandwidth)


def main():
    """
    Calculates and prints the radix and nodes per leaf for different Arista
    spine-leaf configurations with different oversubscription models, using
    tabulate for formatted output and pandas for data storage.
    """

    # Spine switch configurations
    spine_configs = {
        "7804": {
            "line_cards": 4,
            "ports_per_card": 36,
            "port_speed": 800
        },
        "7808": {
            "line_cards": 8,
            "ports_per_card": 36,
            "port_speed": 800
        },
        "7812": {
            "line_cards": 12,
            "ports_per_card": 36,
            "port_speed": 800
        },
        "7816": {
            "line_cards": 16,
            "ports_per_card": 36,
            "port_speed": 800
        },
    }

    # Leaf switch configurations
    leaf_configs = {
        "7060X5": {
            "ports": 64,
            "port_speed": 400,
            "bandwidth": 25600
        },
        "7060X6": {
            "ports": 64,
            "port_speed": 800,
            "bandwidth": 51200
        },
    }

    # Compute node information
    node_bandwidth = 200  # Bandwidth of each compute node's NIC in Gbps

    # Dataframe to store the results
    radix_df = pd.DataFrame(columns=[
        "Leaf", "Spine", "Uplinks", "Workload Ports", "Oversubscription",
        "Max Radix", "Nodes per Leaf"
    ])

    # Calculate Radix and nodes per leaf for different configurations
    for spine_model, spine_config in spine_configs.items():
        spine_ports = spine_config[
            "line_cards"] * spine_config["ports_per_card"]
        for leaf_model, leaf_config in leaf_configs.items():
            leaf_uplink_ports = 8  # Adjust as needed
            for oversubscription, uplink_divisor in [("1:1", 1), ("2:1", 2),
                                                    ("4:1", 4)]:
                uplinks = leaf_uplink_ports // uplink_divisor
                max_radix = calculate_radix(spine_ports, uplinks)
                uplink_bandwidth = uplinks * spine_config[
                    "port_speed"]  # Uplink bandwidth in Gbps
                nodes_per_leaf = calculate_nodes_per_leaf(
                    leaf_config["bandwidth"], node_bandwidth, uplink_bandwidth)
                # Use concat instead of append
                radix_df = pd.concat([
                    radix_df,
                    pd.DataFrame([{
                        "Leaf": leaf_model,
                        "Spine": spine_model,
                        "Uplinks": uplinks,
                        "Workload Ports": leaf_config["ports"] - uplinks,
                        "Oversubscription": oversubscription,
                        "Max Radix": max_radix,
                        "Nodes per Leaf": nodes_per_leaf
                    }])
                ],
                                     ignore_index=True)

    # Print table using tabulate with fancy_grid format
    print(
        tabulate(radix_df,
                 headers='keys',
                 tablefmt="fancy_grid",
                 showindex=False))

    # Considerations
    print("\nConsiderations:")
    print(
        "  - Adjust `leaf_uplink_ports` based on your design and oversubscription requirements."
    )
    print(
        "  - Ensure the spine switch has enough bandwidth for the aggregated traffic."
    )

    # You can now further analyze or use the radix_df DataFrame
    print("\nRadix DataFrame:")
    print(radix_df.head())


if __name__ == "__main__":
    main()

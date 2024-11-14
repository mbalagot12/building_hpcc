import math

def estimate_training_time_and_gpus(
    dataset_size,
    epochs,
    batch_size,
    parameters,
    gpu_type,
    target_training_time_days=None,  # Add target training time in days
    mfu=0.2,
    max_gpus=None  # Add an optional maximum number of GPUs
):
    """
    Estimates the time to train an LLM from scratch, including GPU-specific
    performance, converting the result to days, and estimating the number of
    GPUs required to achieve a target training time.

    Args:
        dataset_size: The size of the training dataset in tokens.
        epochs: The number of training epochs.
        batch_size: The batch size.
        parameters: The number of parameters in the model.
        gpu_type: The type of GPU used for training ("A100", "H100", "V100", "B100", "B200", "Gaudi2", "Gaudi3").
        target_training_time_days: The desired training time in days.
                                     If provided, the function will estimate the
                                     number of GPUs needed to achieve this time.
        mfu: The Model FLOPS Utilization. Defaults to 0.2 (20%).
        max_gpus: An optional maximum number of GPUs to use.

    Returns:
        A dictionary containing:
            - training_time_hours: The estimated training time in hours.
            - training_time_days: The estimated training time in days.
            - num_gpus_needed: The estimated number of GPUs needed to achieve the target training time (if provided).
    """

    # GPU performance in TFLOP/s
    gpu_performance = {
        "L40S": 91.6,
        "A100": 200,
        "H100": 300,  # Approximate
        "V100": 125,
        "B100": 400,  # Approximate
        "B200": 800,  # Approximate
        "Gaudi2": 256,
        "Gaudi3": 1024  # Approximate
    }

    if gpu_type not in gpu_performance:
        raise ValueError("Invalid GPU type.")

    # Calculate the total number of training steps
    steps = (dataset_size * epochs) / batch_size

    # Calculate the total compute required in FLOP/s
    compute = (parameters * steps * 6) / (10**12)  # 6 FLOP/s per parameter

    if target_training_time_days:
        # Calculate the required compute capacity to achieve the target time
        target_time_seconds = target_training_time_days * 24 * 3600
        required_compute_capacity = compute / target_time_seconds

        # Calculate the number of GPUs needed
        num_gpus_needed = math.ceil(required_compute_capacity / (gpu_performance[gpu_type] * mfu))

        if max_gpus is not None:
            num_gpus_needed = min(num_gpus_needed, max_gpus)
    else:
        num_gpus_needed = None  # No target time provided

    # If num_gpus_needed is calculated, use it for time estimation, otherwise, it's not specified
    num_gpus = num_gpus_needed if num_gpus_needed else "Not specified"

    # Calculate the effective compute capacity considering MFU
    effective_compute_capacity = gpu_performance[gpu_type] * num_gpus * mfu

    # Calculate the training time in seconds
    training_time_seconds = compute / effective_compute_capacity

    # Convert the training time to hours and days
    training_time_hours = training_time_seconds / 3600
    training_time_days = training_time_hours / 24

    return {
        "training_time_hours": training_time_hours,
        "training_time_days": training_time_days,
        "num_gpus_needed": num_gpus_needed
    }

# Example usage with target training time and max GPUs
dataset_size = 20e12  # tokens in Trillions
epochs = 1.5
batch_size = 1536
parameters = 405e9  # 405B parameters
gpu_type = "L40S"
target_training_time_days = 10  # Target training time: 10 days
max_gpus = 3000  # Maximum number of GPUs allowed

results = estimate_training_time_and_gpus(
    dataset_size, epochs, batch_size, parameters, gpu_type, target_training_time_days, max_gpus=max_gpus
)

print(f"Estimated training time: {results['training_time_hours']:.2f} hours ({results['training_time_days']:.2f} days)")
print(f"Number of GPUs needed to achieve the target time: {results['num_gpus_needed']}")

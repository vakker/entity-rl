import argparse

import matplotlib.pyplot as plt

from entity_rl.datasets import MOTGraphDataset, MOTVisDataset


def test_vis_dataset(args):
    """Test the MOT vis dataset."""
    print("Testing MOT vis Dataset...")

    dataset = MOTVisDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset info: {dataset.dataset_info}")

    # Generate test samples
    print(f"Generating {args.num_samples} samples...")
    reward_counts = {-1: 0, 1: 0}

    for i in range(args.num_samples):
        img, reward = dataset[i]
        reward_counts[reward] += 1

        if i < 3 and args.save_samples:
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"vis Sample {i}, Reward: {reward}")
            plt.axis("off")
            plt.savefig(f"vis_sample_{i}_reward_{reward}.png")
            plt.close()
            print(f"Saved vis_sample_{i}_reward_{reward}.png")

    print(f"Reward distribution: {reward_counts}")
    collision_rate = reward_counts[-1] / args.num_samples
    print(f"Collision rate: {collision_rate:.2%}")


def test_graph_dataset(args):
    """Test the MOT graph dataset."""
    print("Testing MOT graph Dataset...")

    dataset = MOTGraphDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        use_gt=not args.use_detections,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset info: {dataset.dataset_info}")

    # Generate test samples
    print(f"Generating {args.num_samples} samples...")
    reward_counts = {-1: 0, 1: 0}
    node_counts = []
    edge_counts = []

    for i in range(args.num_samples):
        graph_data, reward = dataset[i]
        reward_counts[reward] += 1
        node_counts.append(graph_data.num_nodes)
        edge_counts.append(graph_data.edge_index.shape[1])

        if i < 3:
            print(
                f"Sample {i}: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges, reward: {reward}"
            )
            print(f"  Node features shape: {graph_data.x.shape}")

    print(f"Reward distribution: {reward_counts}")
    collision_rate = reward_counts[-1] / args.num_samples
    print(f"Collision rate: {collision_rate:.2%}")

    if node_counts:
        print(
            f"Node count - Min: {min(node_counts)}, Max: {max(node_counts)}, Avg: {sum(node_counts)/len(node_counts):.1f}"
        )
        print(
            f"Edge count - Min: {min(edge_counts)}, Max: {max(edge_counts)}, Avg: {sum(edge_counts)/len(edge_counts):.1f}"
        )


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test MOT Datasets")

    # Data arguments
    parser.add_argument(
        "--mot-dirs", nargs="+", required=True, help="List of MOT data directories"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["vis", "graph", "both"],
        default="both",
        help="Which dataset type to test",
    )
    parser.add_argument(
        "--use-detections",
        action="store_true",
        help="Use detection files instead of ground truth",
    )

    # Testing arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate for testing",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save sample images (vis only)",
    )

    # Dataset arguments
    parser.add_argument(
        "--agent-radius", type=float, default=0.02, help="Radius of agent blob in pixels"
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=[500, 500],
        help="Target image size as width height",
    )

    # graph-specific arguments
    parser.add_argument(
        "--max-entities",
        type=int,
        default=20,
        help="Maximum entities per sample (graph only)",
    )
    parser.add_argument(
        "--connect-threshold",
        type=float,
        default=50.0,
        help="Distance threshold for edges (graph only)",
    )

    args = parser.parse_args()

    print(f"Using MOT directories: {args.mot_dirs}")
    print(f"Testing dataset type: {args.dataset_type}")
    print(f"Number of samples: {args.num_samples}")
    print()

    if args.dataset_type in ["vis", "both"]:
        test_vis_dataset(args)
        print()

    if args.dataset_type in ["graph", "both"]:
        test_graph_dataset(args)
        print()

    print("Testing completed successfully!")


if __name__ == "__main__":
    main()


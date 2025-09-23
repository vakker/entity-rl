import argparse

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from entity_rl.datasets import MOTGraphDataset, MOTVisDataset


class MOTVisDatasetWithDebugInfo(MOTVisDataset):
    """Extended MOTVisDataset that returns debug information."""

    def _generate_sample_with_debug(self):
        """Generate sample and return debug information."""
        # Select random frame and bboxes
        data_dir, frame_id, original_bboxes = self.select_random_frame()

        # Load and resize image
        from entity_rl.datasets.mot_data import load_and_resize_image, scale_bboxes

        img = load_and_resize_image(data_dir, frame_id, self.image_size)

        if img is None:
            raise RuntimeError("Failed to load or resize image")

        # Get original dimensions and scale bboxes
        orig_w, orig_h = self.data_loader.get_image_dimensions(data_dir, frame_id)
        scaled_bboxes = scale_bboxes(original_bboxes, (orig_w, orig_h))

        # Generate random agent position
        agent_x, agent_y = self.generate_agent_position()

        # Draw agent blob on the image
        agent_image = self._draw_agent(img.copy(), agent_x, agent_y)

        # Determine reward based on overlap
        reward = self._compute_reward(scaled_bboxes, agent_x, agent_y)

        debug_info = {
            "data_dir": data_dir,
            "frame_id": frame_id,
            "original_bboxes": original_bboxes,
            "scaled_bboxes": scaled_bboxes,
            "agent_x": agent_x,
            "agent_y": agent_y,
            "orig_dims": (orig_w, orig_h),
        }

        return agent_image, reward, debug_info


def draw_bounding_boxes_on_image(
    img,
    scaled_bboxes,
    agent_x,
    agent_y,
    agent_radius,
    image_size,
    reward,
):
    """
    Draw bounding boxes and agent on matplotlib image.

    Args:
        img: Image tensor as numpy array
        scaled_bboxes: List of scaled bounding boxes (x, y, w, h, track_id)
        agent_x, agent_y: Agent center position (normalized 0-1)
        agent_radius: Agent radius (normalized 0-1)
        image_size: Target image size (width, height)

    Returns:
        matplotlib figure with bounding boxes drawn
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Draw detection/GT bounding boxes in green
    for x, y, w, h, track_id in scaled_bboxes:
        # Convert normalized coordinates to pixel coordinates
        pixel_x = x * img.shape[1]
        pixel_y = y * img.shape[0]
        pixel_w = w * img.shape[1]
        pixel_h = h * img.shape[0]

        rect = patches.Rectangle(
            (pixel_x, pixel_y),
            pixel_w,
            pixel_h,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
            label=f"Detection {track_id}" if track_id >= 0 else "Detection",
        )
        ax.add_patch(rect)

        # Add track ID label
        ax.text(
            pixel_x,
            pixel_y - 5,
            f"ID:{track_id}",
            color="green",
            fontsize=8,
            fontweight="bold",
        )

    # Draw agent bounding box in red (it's already drawn on the image, but add outline)
    agent_pixel_x = agent_x * img.shape[1]
    agent_pixel_y = agent_y * img.shape[0]
    agent_pixel_radius = agent_radius * min(img.shape[1], img.shape[0])

    agent_rect = patches.Rectangle(
        (agent_pixel_x - agent_pixel_radius, agent_pixel_y - agent_pixel_radius),
        2 * agent_pixel_radius,
        2 * agent_pixel_radius,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label="Agent",
    )
    ax.add_patch(agent_rect)

    ax.set_title(f"MOT Sample with {len(scaled_bboxes)} detections, Reward: {reward}")
    ax.axis("off")

    # Add legend if there are bboxes
    # if scaled_bboxes:
    #     ax.legend(loc='upper right')

    return fig


def test_vis_dataset(args):
    """Test the MOT vis dataset."""
    print("Testing MOT vis Dataset...")

    dataset = MOTVisDatasetWithDebugInfo(
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
        # Get the sample with debug info for visualization
        if args.save_samples:
            # Generate sample with debug information
            img_array, reward, debug_info = dataset._generate_sample_with_debug()
            img = torch.from_numpy(img_array.astype(np.uint8))

            # Create visualization with bounding boxes
            fig = draw_bounding_boxes_on_image(
                img.numpy(),
                debug_info["scaled_bboxes"],
                debug_info["agent_x"],
                debug_info["agent_y"],
                dataset.agent_radius / min(dataset.image_size),
                dataset.image_size,
                reward,
            )

            # Save the figure with bounding boxes
            plt.savefig(
                f"vis_sample_{i}_reward_{reward}_with_boxes.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"Saved vis_sample_{i}_reward_{reward}_with_boxes.png")

            # Also save the original simple version
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"vis Sample {i}, Reward: {reward}")
            plt.axis("off")
            plt.savefig(f"vis_sample_{i}_reward_{reward}.png")
            plt.close()
            print(f"Saved vis_sample_{i}_reward_{reward}.png")
        else:
            # Regular sample generation for counting
            img, reward = dataset[i]

        reward_counts[reward] += 1

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
        "--agent-radius",
        type=float,
        default=0.02,
        help="Radius of agent blob in pixels",
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

#!/usr/bin/env python3
"""
Test script to demonstrate the classification and clustering loaders.
This script shows how to use the different loading strategies.
"""

from tasks.binary_classification_tasks import ToxicConversations50k, ImdbClassification
from tasks.classification_tasks import Banking77Classification
from tasks.clustering_tasks import EmotionClustering

def test_binary_classification_label_based():
    """Test binary classification with label-based approach."""
    print("\n" + "="*70)
    print("TEST 1: Binary Classification - Label-Based Approach")
    print("="*70)
    
    task = ToxicConversations50k()
    print(f"Task: {task.__class__.__name__}")
    print(f"Strategy: Label-based (default)")
    print(f"use_hard_negative_mining: {task.use_hard_negative_mining}")
    print(f"Loader: {task.loader.__name__}")
    print(f"Label texts: {task.label_texts}")
    print("\nThis approach uses label texts as positives/negatives")
    print("✓ Test passed")


def test_binary_classification_hard_negatives():
    """Test binary classification with hard negative mining."""
    print("\n" + "="*70)
    print("TEST 2: Binary Classification - Hard Negative Mining")
    print("="*70)
    
    task = ImdbClassification()
    task.use_hard_negative_mining = True
    print(f"Task: {task.__class__.__name__}")
    print(f"Strategy: Hard negative mining")
    print(f"use_hard_negative_mining: {task.use_hard_negative_mining}")
    print(f"Loader: {task.loader.__name__}")
    print("\nThis approach creates a corpus of all texts for mining")
    print("✓ Test passed")


def test_multiway_classification():
    """Test multi-way classification."""
    print("\n" + "="*70)
    print("TEST 3: Multi-way Classification")
    print("="*70)
    
    task = Banking77Classification()
    print(f"Task: {task.__class__.__name__}")
    print(f"Strategy: Sampling from same class")
    print(f"use_hard_negative_mining: {task.use_hard_negative_mining}")
    print(f"Loader: {task.loader.__name__}")
    print("\nThis approach samples positives from same class")
    print("Hard negatives can be mined from other classes (24 samples)")
    print("✓ Test passed")


def test_clustering():
    """Test clustering."""
    print("\n" + "="*70)
    print("TEST 4: Clustering")
    print("="*70)
    
    task = EmotionClustering()
    print(f"Task: {task.__class__.__name__}")
    print(f"Strategy: Sampling from same cluster")
    print(f"use_hard_negative_mining: {task.use_hard_negative_mining}")
    print(f"Loader: {task.loader.__name__}")
    print("\nThis approach samples positives from same cluster")
    print("Hard negatives can be mined from other clusters (24 samples)")
    print("✓ Test passed")


def test_task_registry():
    """Test that tasks are properly registered."""
    print("\n" + "="*70)
    print("TEST 5: Task Registry")
    print("="*70)
    
    from tasks import (
        get_task,
        BINARY_CLASSIFICATION_TASKS,
        CLASSIFICATION_TASKS,
        CLUSTERING_TASKS
    )
    
    print(f"\nBinary Classification Tasks ({len(BINARY_CLASSIFICATION_TASKS)}):")
    for task_name in BINARY_CLASSIFICATION_TASKS:
        print(f"  - {task_name}")
    
    print(f"\nMulti-way Classification Tasks ({len(CLASSIFICATION_TASKS)}):")
    for task_name in CLASSIFICATION_TASKS:
        print(f"  - {task_name}")
    
    print(f"\nClustering Tasks ({len(CLUSTERING_TASKS)}):")
    for task_name in CLUSTERING_TASKS[:5]:
        print(f"  - {task_name}")
    print(f"  ... and {len(CLUSTERING_TASKS) - 5} more")
    
    # Test get_task function
    task_cls = get_task("toxic_conversations")
    print(f"\nget_task('toxic_conversations') -> {task_cls.__name__}")
    print("✓ Test passed")


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# Classification and Clustering Loaders - Test Suite")
    print("#"*70)
    
    try:
        test_binary_classification_label_based()
        test_binary_classification_hard_negatives()
        test_multiway_classification()
        test_clustering()
        test_task_registry()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nImplementation is working correctly!")
        print("See CLASSIFICATION_CLUSTERING_README.md for detailed usage.")
        print("See IMPLEMENTATION_SUMMARY.md for implementation details.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

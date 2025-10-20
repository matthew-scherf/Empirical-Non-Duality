"""
TUOS-SGNA: MNIST Evaluation
============================
Testing substrate-grounded neural architecture on real-world image classification.

This extends the proof-of-concept to MNIST (28x28 grayscale digit recognition)
to demonstrate the framework scales beyond synthetic toy tasks while maintaining
axiom satisfaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# Import SGNA components
from sgna import (
    SubstrateLayer, GaugeInvariantClassifier, PhenomenonGraph,
    TUOSConfig, AdaptiveTrainer, MetricsLogger, substrate_context
)


def load_mnist(batch_size=64, data_dir='./data'):
    """Load MNIST dataset with standard preprocessing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader


def evaluate(substrate, classifier, test_loader, device):
    """Evaluate model on test set with detailed metrics."""
    substrate.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    insep_scores = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            presentations = substrate.present(images, mode='data')
            predictions = classifier.predict(presentations)
            
            _, predicted = predictions['substrate_prediction'].max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            insep_scores.extend(
                predictions['inseparability'].cpu().tolist()
            )
    
    accuracy = correct / total
    mean_insep = sum(insep_scores) / len(insep_scores)
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'mean_inseparability': mean_insep,
        'insep_scores': insep_scores
    }


def train_mnist_sgna(config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train SGNA on MNIST with full axiom enforcement.
    
    Args:
        config: TUOSConfig or None for defaults
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with trained models and results
    """
    
    if config is None:
        config = TUOSConfig(
            substrate_dim=512,  # Larger for MNIST
            num_classes=10,
            input_dim=784,      # 28x28 flattened
            num_epochs=20,
            learning_rate=0.001,
            batch_size=128,
            n_phenomena=100,    # Fewer phenomena for efficiency
            edge_probability=0.03,
            use_adaptive_weights=True,
            use_parametric=True,
            use_harmonic=False,  # Disable for speed
            use_dependent_arising=False,
            target_inseparability=0.80,  # Slightly lower target
            target_temporal_violation_rate=0.0,
            log_interval=1
        )
    
    print("=" * 70)
    print("TUOS-SGNA: MNIST Evaluation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: MNIST (28x28 grayscale digits)")
    print(f"  Training samples: 60,000")
    print(f"  Test samples: 10,000")
    print(f"  Substrate dimension: {config.substrate_dim}")
    print(f"  Number of classes: {config.num_classes}")
    print(f"  Phenomena count: {config.n_phenomena}")
    print(f"  Training epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device: {device}")
    print()
    
    # Load MNIST
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(
        batch_size=config.batch_size
    )
    print(f"Loaded {len(train_loader.dataset)} training samples")
    print(f"Loaded {len(test_loader.dataset)} test samples\n")
    
    with substrate_context():
        # Initialize architecture
        substrate = SubstrateLayer(config).to(device)
        substrate.register_presentation_mode('data', config.input_dim)
        if config.use_parametric:
            substrate.register_presentation_mode('parametric', config.input_dim)
        
        classifier = GaugeInvariantClassifier(substrate).to(device)
        
        graph = PhenomenonGraph(config)
        graph.build_random_dag()
        graph.T = graph.T.to(device)
        
        trainer = AdaptiveTrainer(substrate, classifier, graph, config)
        
        # Setup optimizer with proper handling of adaptive weights
        param_dict = {}
        for p in substrate.parameters():
            param_dict[id(p)] = p
        for p in classifier.parameters():
            param_dict[id(p)] = p
        param_dict[id(graph.T)] = graph.T
        
        # Only add lambda parameters if they are actual Parameters (not floats)
        if config.use_adaptive_weights:
            if isinstance(trainer.lambda_insep, nn.Parameter):
                param_dict[id(trainer.lambda_insep)] = trainer.lambda_insep
            if isinstance(trainer.lambda_time, nn.Parameter):
                param_dict[id(trainer.lambda_time)] = trainer.lambda_time
        
        params = list(param_dict.values())
        
        # Verify all params are leaf tensors
        params = [p for p in params if p.is_leaf]
        
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )
        
        # Training loop
        print("Training with integrated axiom constraints...\n")
        
        for epoch in range(config.num_epochs):
            substrate.train()
            classifier.train()
            
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            epoch_insep = []
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                presentations = substrate.present(images, mode='data')
                predictions = classifier.predict(presentations)
                
                # Compute loss
                loss, metrics = trainer.compute_loss(
                    presentations, predictions, labels
                )
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                _, predicted = predictions['substrate_prediction'].max(1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_total += labels.size(0)
                epoch_insep.append(metrics['mean_inseparability'])
            
            scheduler.step()
            
            # Adapt weights with proper metrics
            num_violations = len(graph.get_violation_details())
            violation_rate = num_violations / len(graph.edges) if len(graph.edges) > 0 else 0.0
            
            trainer.adapt_weights({
                'mean_inseparability': sum(epoch_insep) / len(epoch_insep),
                'num_violations': num_violations,
                'violation_rate': violation_rate
            })
            
            # Evaluate on test set
            test_results = evaluate(substrate, classifier, test_loader, device)
            
            # Print progress
            train_acc = epoch_correct / epoch_total
            train_insep = sum(epoch_insep) / len(epoch_insep)
            viol_count = len(graph.get_violation_details())
            
            print(f"Epoch {epoch+1:2d}/{config.num_epochs}: "
                  f"Train Acc={train_acc:.4f}, "
                  f"Test Acc={test_results['accuracy']:.4f}, "
                  f"Insep={train_insep:.3f}, "
                  f"Viol={viol_count}")
        
        # Final comprehensive evaluation
        print("\n" + "="*70)
        print("FINAL MNIST EVALUATION")
        print("="*70)
        
        final_test = evaluate(substrate, classifier, test_loader, device)
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy: {final_test['accuracy']:.4f} ({final_test['correct']}/{final_test['total']})")
        print(f"  Error Rate: {(1-final_test['accuracy'])*100:.2f}%")
        
        print(f"\nInseparability Statistics:")
        insep_tensor = torch.tensor(final_test['insep_scores'])
        print(f"  Mean: {insep_tensor.mean().item():.4f}")
        print(f"  Std:  {insep_tensor.std().item():.4f}")
        print(f"  Min:  {insep_tensor.min().item():.4f}")
        print(f"  Max:  {insep_tensor.max().item():.4f}")
        print(f"  Samples > 0.8: {(insep_tensor > 0.8).sum().item()}/{len(insep_tensor)}")
        print(f"  Samples > 0.5: {(insep_tensor > 0.5).sum().item()}/{len(insep_tensor)}")
        
        print(f"\nTemporal Structure:")
        violations = graph.get_violation_details()
        print(f"  Total edges: {len(graph.edges)}")
        print(f"  Violations: {len(violations)}/{len(graph.edges)} "
              f"({len(violations)/len(graph.edges)*100:.2f}%)")
        
        print(f"\nSubstrate Properties:")
        print(f"  Omega norm: {substrate.omega.norm().item():.4f}")
        print(f"  Dimensions: {config.substrate_dim}")
        
        print("\n" + "="*70)
        
        # Summary
        if final_test['accuracy'] >= 0.95 and insep_tensor.mean().item() >= 0.7:
            print("\nSUMMARY: Successfully scaled to MNIST!")
            print(f"Achieved {final_test['accuracy']:.1%} accuracy while maintaining")
            print(f"{insep_tensor.mean().item():.1%} mean inseparability.")
            print("Metaphysical constraints remain effective on real-world task.")
        else:
            print("\nSUMMARY: MNIST results show trade-offs.")
            print(f"Accuracy: {final_test['accuracy']:.1%}, Inseparability: {insep_tensor.mean().item():.1%}")
        
        print("="*70 + "\n")
        
        return {
            'substrate': substrate,
            'classifier': classifier,
            'graph': graph,
            'config': config,
            'test_results': final_test,
            'train_loader': train_loader,
            'test_loader': test_loader
        }


def quick_baseline_comparison():
    """
    Train a standard MLP on MNIST for comparison.
    """
    print("=" * 70)
    print("Baseline MLP (No Constraints)")
    print("=" * 70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simple MLP
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    ).to(device)
    
    train_loader, test_loader = load_mnist(batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 5 epochs (quick comparison)
    for epoch in range(5):
        model.train()
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
        
        print(f"Epoch {epoch+1}: Train Acc={correct/total:.4f}, "
              f"Test Acc={test_correct/test_total:.4f}")
    
    print(f"\nBaseline achieves ~{test_correct/test_total:.1%} without constraints\n")
    return test_correct/test_total


if __name__ == "__main__":
    print("\nTUOS-SGNA MNIST Evaluation")
    print("Testing substrate-grounded architecture on real image classification\n")
    
    # Optional: Run baseline first for comparison
    if len(sys.argv) > 1 and sys.argv[1] == '--with-baseline':
        baseline_acc = quick_baseline_comparison()
    
    # Train SGNA on MNIST
    results = train_mnist_sgna()
    
    print("\nEvaluation complete!")
    print("\nNext steps:")
    print("  - Try different substrate dimensions")
    print("  - Enable parametric mode for higher inseparability")
    print("  - Test on CIFAR-10 (3x32x32 color images)")
    print("  - Compare to baseline MLPs and CNNs")
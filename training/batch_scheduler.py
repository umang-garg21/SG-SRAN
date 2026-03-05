# -*-coding:utf-8 -*-
"""
File:        batch_scheduler.py
Created at:  2025/11/03
Author:      Warren Zamudio
Description: Dynamic batch size scheduler for progressive training.
"""


class BatchSizeScheduler:
    """
    Dynamic batch size scheduler that adjusts batch size based on epoch.
    
    Schedule:
    - Epochs 0-499: batch_size = 64
    - Epochs 500-599: batch_size = 32 (divide by 2)
    - Epochs 600-699: batch_size = 16 (divide by 2)
    - Epochs 700-799: batch_size = 8 (divide by 2)
    - Epochs 800-899: batch_size = 4 (divide by 2, reached minimum)
    - Epochs 900-999: batch_size = 2 (alternate)
    - Epochs 1000-1099: batch_size = 4 (alternate)
    - ... continue alternating between 4 and 2 every 100 epochs until epoch 2000
    """
    
    def __init__(self, initial_batch_size=64, min_batch_size=4):
        """
        Parameters
        ----------
        initial_batch_size : int
            Starting batch size (default 64)
        min_batch_size : int
            Minimum batch size before alternating phase (default 4)
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        
    def get_batch_size(self, epoch):
        """
        Get the batch size for a given epoch.
        
        Parameters
        ----------
        epoch : int
            Current training epoch (0-indexed)
            
        Returns
        -------
        int
            Batch size to use for this epoch
        """
        # Phase 1: Keep initial batch size from epoch 0-499 (500 epochs total)
        if epoch < 500:
            return self.initial_batch_size
        
        # Phase 2: Starting at epoch 500, divide by 2 every 100 epochs
        # Epoch 500-599: divide to 32
        # Epoch 600-699: divide to 16
        # Epoch 700-799: divide to 8
        # Epoch 800-899: divide to 4 (reached minimum)
        epochs_since_500 = epoch - 500
        division_periods = (epochs_since_500 // 100) + 1  # +1 because first division happens at 500
        
        # Calculate current batch size after divisions
        batch_size = self.initial_batch_size // (2 ** division_periods)
        
        # Phase 3: Once we reach min_batch_size, alternate between min and min/2
        if batch_size <= self.min_batch_size:
            # Calculate how many divisions it takes to reach min from initial
            import math
            divisions_to_min = int(math.log2(self.initial_batch_size / self.min_batch_size))
            epoch_at_min = 500 + ((divisions_to_min - 1) * 100)  # -1 because first division is at 500
            
            # Calculate which 100-epoch block we're in after reaching min
            epochs_since_min = epoch - epoch_at_min
            block_number = epochs_since_min // 100
            
            # First block (0): use min_batch_size (4)
            # Second block (1): use min_batch_size / 2 (2)
            # Third block (2): use min_batch_size (4)
            # Fourth block (3): use min_batch_size / 2 (2)
            # Continue alternating...
            if block_number % 2 == 0:
                return self.min_batch_size  # Blocks 0, 2, 4, ...: use 4
            else:
                return max(self.min_batch_size // 2, 1)  # Blocks 1, 3, 5, ...: use 2
        
        return max(batch_size, 1)  # Safety: never go below 1
    
    def get_schedule_summary(self, max_epochs=2000):
        """
        Print a summary of the batch size schedule.
        
        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs to summarize
        """
        print("\n" + "="*80)
        print("BATCH SIZE SCHEDULE")
        print("="*80)
        
        prev_bs = None
        epoch_ranges = []
        start_epoch = 0
        
        for epoch in range(max_epochs + 1):
            current_bs = self.get_batch_size(epoch)
            
            if prev_bs is not None and current_bs != prev_bs:
                epoch_ranges.append((start_epoch, epoch - 1, prev_bs))
                start_epoch = epoch
            
            prev_bs = current_bs
        
        # Add final range
        if prev_bs is not None:
            epoch_ranges.append((start_epoch, max_epochs, prev_bs))
        
        for start, end, bs in epoch_ranges:
            print(f"Epochs {start:4d}-{end:4d}: batch_size = {bs:2d}")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    # Test the scheduler
    scheduler = BatchSizeScheduler(initial_batch_size=64, min_batch_size=4)
    scheduler.get_schedule_summary(max_epochs=2000)
    
    # Verify specific epochs
    test_epochs = [0, 499, 500, 599, 600, 1100, 1199, 1200, 1299, 1300, 1999, 2000]
    print("\nSpot checks:")
    for epoch in test_epochs:
        bs = scheduler.get_batch_size(epoch)
        print(f"Epoch {epoch:4d}: batch_size = {bs:2d}")

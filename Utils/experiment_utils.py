import pygame
import numpy as np
import random
import pickle
from collections import deque
import os

class RollingScaler:
    def __init__(self, window_size=100, save_path=None):
        """
        A rolling Z-score normalizer that updates mean and std dynamically.
        
        - `window_size`: The max number of windows to store.
        - `save_path`: Optional path to save/load rolling stats between sessions.
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.save_path = save_path  # File to save/load rolling mean & std

    def update(self, new_data):
        """Store new EEG data in the rolling window buffer."""
        self.buffer.append(new_data)

    def transform(self, new_data):
        """Apply z-score normalization using a single mean/std across all stored EEG values."""
        if len(self.buffer) == 0:
            mean = np.mean(new_data)  # Compute single mean over all values
            std = np.std(new_data) + 1e-8  # Compute single std over all values
        else:
            past_data = np.concatenate(self.buffer, axis=1).flatten()  # Flatten to (total_values,)
            mean = np.mean(past_data)  # Compute single mean across all data
            std = np.std(past_data)  # Compute single std across all data

        return (new_data - mean) / std


    def save(self):
        """Save the latest rolling buffer for continuity across runs."""
        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(list(self.buffer), f)  # Convert deque to list before saving

    def load(self):
        """Load previously saved rolling buffer if available."""
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                self.buffer = deque(pickle.load(f), maxlen=self.window_size)
                print(f"✅ Loaded rolling stats from: {self.save_path}")


class LeakyIntegrator:
    def __init__(self, alpha=0.95):
        """
        Initializes the leaky integrator.
        
        :param alpha: Leak factor (0 < alpha < 1), where higher values retain past values longer.
        """
        self.alpha = alpha
        self.accumulated_probability = 0.5  # Initial probability

    def update(self, new_probability):
        """
        Updates the accumulated probability using a leaky integration method.

        :param new_probability: The new probability input from the classifier.
        :return: The updated accumulated probability.
        """
        self.accumulated_probability = self.alpha * self.accumulated_probability + (1 - self.alpha) * new_probability
        return self.accumulated_probability



def generate_trial_sequence(total_trials=30, max_repeats=3):
    trials = [0] * (total_trials // 2) + [1] * (total_trials // 2)
    random.shuffle(trials)
    fixed_trials = []
    for trial in trials:
        if len(fixed_trials) >= max_repeats and all(t == trial for t in fixed_trials[-max_repeats:]):
            alternatives = [t for t in set([0, 1]) if t != trial]
            random.shuffle(alternatives)
            trial = alternatives[0]
        fixed_trials.append(trial)
    return fixed_trials


# for ErrP
def generate_trial_sequence_with_errp(total_trials=45, max_repeats=3):
    # Calculate how many of each type (0, 1, 2) are required.
    n_per_type = total_trials // 3
    # Remaining counts for each trial type.
    counts = {0: n_per_type, 1: n_per_type, 2: n_per_type}
    
    # Backtracking helper that builds the sequence recursively.
    def backtrack(seq, counts):
        # Base case: If the sequence is complete, return it.
        if len(seq) == total_trials:
            return seq
        
        # Build a list of valid trial types
        valid = []
        for t in [0, 1, 2]:
            if counts[t] > 0:
                # Check if adding this type would break the repeat rule.
                if len(seq) < max_repeats or not all(prev == t for prev in seq[-max_repeats:]):
                    valid.append(t)
                    
        # Randomize the order of trying valid options to maintain randomness.
        random.shuffle(valid)
        for t in valid:
            # Choose t and update the counts.
            seq.append(t)
            counts[t] -= 1
            # Continue building the sequence recursively.
            result = backtrack(seq, counts)
            if result is not None:
                return result   # Valid complete sequence found.
            # Backtrack: remove the last choice and restore the count.
            seq.pop()
            counts[t] += 1
        return None  # No valid extension from this state.
    
    # Start the backtracking process with an empty sequence.
    return backtrack([], counts)


def display_multiple_messages_with_udp(messages, colors, offsets, duration=13, udp_messages=None, udp_socket=None, udp_ip=None, udp_port=None):
    font = pygame.font.SysFont(None, 72)
    end_time = pygame.time.get_ticks() + duration * 1000

    udp_sent = False
    while pygame.time.get_ticks() < end_time:
        pygame.display.get_surface().fill((0, 0, 0))
        for i, text in enumerate(messages):
            message = font.render(text, True, colors[i])
            pygame.display.get_surface().blit(
                message,
                (pygame.display.get_surface().get_width() // 2 - message.get_width() // 2,
                 pygame.display.get_surface().get_height() // 2 - message.get_height() // 2 + offsets[i])
            )
        pygame.display.flip()
        if udp_messages and not udp_sent:
            for msg in udp_messages:
                udp_socket.sendto(msg.encode('utf-8'), (udp_ip, udp_port))
                print(f"Sent UDP message to {udp_ip}:{udp_port}: {msg}")
            udp_sent = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        pygame.time.Clock().tick(60)
import torch
from dqn import DQN


class AIPlayer:
    def __init__(
        self,
        input_shape,
        n_actions,
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = DQN(input_shape, n_actions).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.n_actions = n_actions

    def select_action(self, state, legal_actions):
        with torch.no_grad():
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.model(state)


            mask = torch.full((1, self.n_actions), float("-inf"), device=self.device)
            mask[0, legal_actions] = 0


            masked_q_values = q_values + mask


            return masked_q_values.max(1)[1].view(1, 1)

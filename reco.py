import numpy as np
from typing import List
import time

def safelog(x):
    """
    Function to safely compute the logarithm of x. If x is less than a very small positive number (EPS), 
    the logarithm of EPS is returned to avoid division by zero errors.
    """
    EPS = 1e-60
    return np.log(np.maximum(x, EPS))


class ReCoTrainer:
    """
    This class implements the ReCoTrainer, a model trainer for reference games. 
    """
    def __init__(
        self,
        states: List[str],
        options: List[str],
        state_prior: np.ndarray,
        generator_priors: np.ndarray,
        discriminator_priors: np.ndarray,
        generator_lambda_param: np.ndarray,
        discriminator_lambda_param: np.ndarray,
        alternating: bool = True,
        seed: int = 0
    ):
        """
        Initializes the FastPiKLReferenceGameTrainer with the provided parameters.
        Parameters:
        states (List[str]): List of possible states in the game.
        options (List[str]): List of possible options in the game.
        state_prior (np.ndarray): Prior probabilities of the states.
        generator_priors (np.ndarray): Prior probabilities of the generator's actions.
        discriminator_priors (np.ndarray): Prior probabilities of the discriminator's actions.
        generator_lambda_param (np.ndarray): Lambda parameter for the generator's strategy.
        discriminator_lambda_param (np.ndarray): Lambda parameter for the discriminator's strategy.
        alternating (bool, optional): If True, the generator and discriminator alternate turns. Defaults to True.
        seed (int, optional): Seed for random number generation. Defaults to 0.
        """
        assert generator_lambda_param.shape == discriminator_lambda_param.shape
        assert len(generator_lambda_param.shape) == 1
        assert len(discriminator_lambda_param.shape) == 1
        self.generator_lambda_param = generator_lambda_param
        self.discriminator_lambda_param = discriminator_lambda_param
        self.generator_eta = 0.1
        self.discriminator_eta = 0.1
        self.alternating = alternating

        self.state_prior = state_prior
        self.generator_priors = generator_priors
        self.discriminator_priors = discriminator_priors

        assert len(states) == len(state_prior)
        assert len(states) == generator_priors.shape[0]
        assert len(options) == generator_priors.shape[1]
        assert len(options) == discriminator_priors.shape[0]
        assert len(states) == discriminator_priors.shape[1]

        self.states = states
        self.options = options

        self.generator_cum_val_matrix = np.zeros((len(self.generator_lambda_param), len(self.states), len(self.options)))
        self.discriminator_cum_val_matrix = np.zeros(
            (len(self.discriminator_lambda_param), len(self.options), len(self.states))
        )

        if seed:
            np.random.seed(seed)
            self.generator_cum_val_matrix = np.random.exponential(scale=1, size=self.generator_cum_val_matrix.shape)
            self.discriminator_cum_val_matrix = np.random.exponential(scale=1, size=self.discriminator_cum_val_matrix.shape)

            self.generator_cum_val_matrix /= self.generator_cum_val_matrix.sum(-1)[:,:,np.newaxis]
            self.discriminator_cum_val_matrix /= self.discriminator_cum_val_matrix.sum(-1)[:,:,np.newaxis]

        self.generator_cum_strategy_matrix = np.zeros(
            (len(self.generator_lambda_param), len(self.states), len(self.options))
        )
        self.discriminator_cum_strategy_matrix = np.zeros(
            (len(self.discriminator_lambda_param), len(self.options), len(self.states))
        )

        self.current_iteration = 0

    def get_discriminator_reco_strategies(self) -> np.array:
        """
        This method calculates the discriminator's strategies using the PiKL (Policy regularized Hedge) algorithm, 
        introduced by Jacob et al. The strategy is computed based on the cumulative value matrix, the current 
        iteration, the eta parameter, and the lambda parameter of the discriminator. The result is then normalized 
        using the softmax function to ensure that the strategy values are probabilities that sum to 1.
        Returns:
            np.array: A 3D numpy array representing the discriminator's strategies. The dimensions correspond to 
                      the lambda parameter, options, and states respectively.
        """
        strategies = (
            self.discriminator_cum_val_matrix * self.discriminator_eta
            + self.current_iteration
            * self.discriminator_eta
            * self.discriminator_lambda_param.reshape(-1, 1, 1)
            * safelog(self.discriminator_priors)
        ) / (
            1 + self.current_iteration * self.discriminator_eta * self.discriminator_lambda_param.reshape(-1,1,1)
        )

        return np.exp(strategies) / np.sum(np.exp(strategies), axis=2)[:,:,np.newaxis]

    def get_generator_reco_strategies(self):
        """
        This method calculates the generator's strategies using the PiKL (Policy regularized Hedge) algorithm, 
        introduced by Jacob et al. The strategy is computed based on the cumulative value matrix, the current 
        iteration, the eta parameter, and the lambda parameter of the generator. The result is then normalized 
        using the softmax function to ensure that the strategy values are probabilities that sum to 1.
        Returns:
            np.array: A 3D numpy array representing the generator's strategies. The dimensions correspond to 
                      the lambda parameter, states, and options respectively.
        """
        strategies = (
            self.generator_cum_val_matrix * self.generator_eta
            + self.current_iteration
            * self.generator_eta
            * self.generator_lambda_param.reshape(-1, 1, 1)
            * safelog(self.generator_priors)
        ) / (
            1 + self.current_iteration * self.generator_eta * self.generator_lambda_param.reshape(-1,1,1)
        )

        return np.exp(strategies) / np.sum(np.exp(strategies), axis=2)[:,:,np.newaxis]

    def get_generator_average_strategy(self):
        """
        This method calculates and returns the average strategy of the generator over all iterations. 
        It asserts that the current iteration is greater than 0 to ensure that the division operation is valid.
        Returns:
            np.array: A 3D numpy array representing the average strategy of the generator. The dimensions correspond to 
                      the lambda parameter, states, and options respectively.
        """
        assert self.current_iteration > 0
        return self.generator_cum_strategy_matrix / self.current_iteration

    def get_discriminator_average_strategy(self):
        """
        This method calculates and returns the average strategy of the discriminator over all iterations. 
        It asserts that the current iteration is greater than 0 to ensure that the division operation is valid.
        Returns:
            np.array: A 3D numpy array representing the average strategy of the discriminator. The dimensions correspond to 
                      the lambda parameter, options, and states respectively.
        """
        assert self.current_iteration > 0
        return self.discriminator_cum_strategy_matrix / self.current_iteration

    def get_regularized_evs(self):
        """
        This method calculates and returns the regularized expected values for the generator and discriminator. 
        It first retrieves the generator and discriminator strategies, then computes the base utility, the divergence 
        for the generator and discriminator, and finally returns the difference between the base utility and the sum 
        of the divergences.
        Returns:
            np.array: A 1D numpy array representing the regularized expected values for the generator and discriminator.
        """
        x = self.get_generator_reco_strategies()
        y = self.get_discriminator_reco_strategies()

        base_utility = (
            (x * y.transpose(0,2,1))
            .sum(-1)
            @ self.state_prior
        )

        div_S = (
            (
                (safelog(x) - safelog(self.generator_priors))
                * x
            ).sum((1,2))
            * self.generator_lambda_param
        )

        div_L = (
            (
                (safelog(y) - safelog(self.discriminator_priors))
                * y
            ).sum((1,2))
            * self.discriminator_lambda_param
        )

        return base_utility - div_S - div_L

    def train(self, num_iterations) -> List[np.ndarray]:
        """
        This method trains the model for the specified number of iterations. In each iteration, it computes 
        the discriminator and generator strategies, updates the cumulative value matrices for the generator and discriminator, 
        updates the cumulative strategy matrices for the generator and discriminator, and increments the current iteration. 
        After all iterations, it returns the average strategies for the generator and discriminator.
        Parameters:
            num_iterations (int): The number of iterations to train the model.
        Returns:
            List[np.ndarray]: A list containing two 3D numpy arrays representing the average strategies for the 
                              generator and discriminator respectively. The dimensions correspond to the lambda parameter, 
                              states/options, and options/states respectively.
        """
        for _ in range(num_iterations):
            if self.alternating:
                discriminator_strategies = self.get_discriminator_reco_strategies()
                generator_utilities = (
                    discriminator_strategies.transpose(0,2,1) * self.state_prior[:, np.newaxis]
                )
                self.generator_cum_val_matrix += generator_utilities

                generator_strategies = self.get_generator_reco_strategies()
                discriminator_utilities = generator_strategies.transpose(0,2,1) * self.state_prior
                self.discriminator_cum_val_matrix += discriminator_utilities
            else:
                discriminator_strategies = self.get_discriminator_reco_strategies()
                generator_utilities = (
                    discriminator_strategies.transpose(0,2,1) * self.state_prior[:, np.newaxis]
                )

                generator_strategies = self.get_generator_reco_strategies()
                discriminator_utilities = generator_strategies.transpose(0,2,1) * self.state_prior

                self.generator_cum_val_matrix += generator_utilities
                self.discriminator_cum_val_matrix += discriminator_utilities

            self.generator_cum_strategy_matrix += generator_strategies
            self.discriminator_cum_strategy_matrix += discriminator_strategies
            self.current_iteration += 1
        
        return self.get_generator_average_strategy(), self.get_discriminator_average_strategy()


if __name__ == "__main__":
    generator_states = ["blue-square", "blue-circle"]
    generator_actions = ["square", "circle", "green", "blue"]
    discriminator_actions = ["blue-square", "blue-circle"]
    generator_prior_dict = {
        "blue-square": np.array([0.3863, 0.5288, 0.5385, 0.5132]),
        "blue-circle": np.array([0.6137, 0.4712, 0.4615, 0.4868]),
    }
    discriminator_prior_dict = {
        "square": np.array([0.2446, 0.2585]),
        "circle": np.array([0.2457, 0.2569]),
        "green": np.array([0.2408, 0.2645]),
        "blue": np.array([0.2689, 0.2201]),
    }

    state_prior = np.ones(len(generator_states)) / len(generator_states)
    generator_priors = np.vstack([policy for k, policy in generator_prior_dict.items()])
    discriminator_priors = np.vstack([policy for k, policy in discriminator_prior_dict.items()])

    discriminator_lambda_values = [1e-3]
    generator_lambda_values = [1e-3]

    lambda_S, lambda_L = np.meshgrid(discriminator_lambda_values, generator_lambda_values)
    lambda_S = lambda_S.reshape(-1)
    lambda_L = lambda_L.reshape(-1)

    reco_trainer = ReCoTrainer(
        generator_states,
        generator_actions,
        state_prior,
        generator_priors,
        discriminator_priors,
        generator_lambda_param=lambda_S,
        discriminator_lambda_param=lambda_L,
        alternating=False,
        seed = 0
    )

    start_time = time.time()

    generator_avg_strategy, discriminator_avg_strategy = reco_trainer.train(5000)
    print(generator_avg_strategy, discriminator_avg_strategy)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    print(reco_trainer.get_regularized_evs())
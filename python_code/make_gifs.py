import imageio
import matplotlib.pyplot as plt
import numpy as np


def make_beta_gif(beta_binom_model, n_iter_plot=50, gif_name='beta_binomial'):
    filenames = []
    probability_range = np.linspace(beta_binom_model.proposals.min(), beta_binom_model.proposals.max(), 100)
    posteriors = [beta_binom_model.calculate_posterior(probability) for probability in probability_range]
    for it in range(n_iter_plot):
        plt.figure(figsize=(8, 4))
        plt.plot(probability_range, posteriors, lw=3, alpha=.5)
        probabilities_heads = beta_binom_model.probability_heads_traces[:it + 1]
        posteriors_probabilities_heads = beta_binom_model.posteriors[:it + 1]
        plt.scatter(probabilities_heads, posteriors_probabilities_heads, color='black', marker='+', s=150, zorder=3)
        plt.plot(probabilities_heads, posteriors_probabilities_heads, color='black', alpha=.3, lw=3)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('probability of heads', fontsize=15)
        plt.ylabel('posterior density', fontsize=15)
        filename = f'beta_videos/{3 * it + 1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        filenames.append(filename)
        plt.figure(figsize=(8, 4))
        plt.plot(probabilities_heads, posteriors_probabilities_heads, color='black', alpha=.3, lw=3)
        plt.plot(probability_range, posteriors, lw=3, alpha=.5)
        proposal = beta_binom_model.proposals[it]
        posterior_proposal = beta_binom_model.calculate_posterior(proposal)
        plt.plot([beta_binom_model.probability_heads_traces[it], proposal], [beta_binom_model.posteriors[it],
                                                                             posterior_proposal], lw=3, alpha=.3,
                 linestyle='--', color='black')
        plt.scatter(probabilities_heads, posteriors_probabilities_heads, color='black', marker='+', s=150, zorder=3)
        plt.scatter(proposal, posterior_proposal, color='black', marker='+', s=150, zorder=3)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('probability of heads', fontsize=15)
        plt.ylabel('posterior density', fontsize=15)
        filename = f'beta_videos/{3 * it + 2}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        filenames.append(filename)
        plt.figure(figsize=(8, 4))
        plt.plot(probabilities_heads, posteriors_probabilities_heads, color='black', alpha=.3, lw=3)
        plt.plot(probability_range, posteriors, lw=3, alpha=.5)
        if beta_binom_model.accept_array[it]:
            plt.scatter(proposal, posterior_proposal, color='green', marker='+', s=150, zorder=3)
        else:
            plt.scatter(proposal, posterior_proposal, color='red', marker='+', s=150, zorder=3)
        plt.plot([beta_binom_model.probability_heads_traces[it], proposal]
                 , [beta_binom_model.posteriors[it], posterior_proposal], lw=3, alpha=.3, linestyle='--', color='black')
        plt.scatter(probabilities_heads, posteriors_probabilities_heads, color='black', marker='+', s=150, zorder=3)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('probability of heads', fontsize=15)
        plt.ylabel('posterior density', fontsize=15)
        filename = f'beta_videos/{3 * it + 3}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        filenames.append(filename)
    save_gif(filenames, gif_name)


def make_logistic_regression_gif(logistic_model, n_iter_plot=50, n_points=50, gif_name='logistic_regression'):
    filenames = []
    beta_0_grid = np.linspace(logistic_model.proposals[:, 0].min() - 0.1, logistic_model.proposals[:, 0].max() + 0.1,
                              n_points)
    beta_1_grid = np.linspace(logistic_model.proposals[:, 1].min() - 0.1, logistic_model.proposals[:, 1].max() + 0.1,
                              n_points)
    betas_0_range, betas_1_range = np.meshgrid(beta_0_grid, beta_1_grid)
    betas_0_range, betas_1_range = betas_0_range.flatten(), betas_1_range.flatten()
    betas_grid = np.array([betas_0_range, betas_1_range])
    posteriors_betas = [logistic_model.calculate_posterior(betas_grid[:, n]) for n in range(len(betas_0_range))]
    for it in range(n_iter_plot):
        plt.figure(facecolor='white')
        plt.grid(False)
        plt.contourf(beta_0_grid, beta_1_grid, np.array(np.log(posteriors_betas)).reshape(n_points, n_points),
                     cmap='coolwarm', levels=150, alpha=.2)
        plt.scatter(logistic_model.betas_traces[:it + 1, 0], logistic_model.betas_traces[:it + 1, 1], color='black',
                    marker='+', s=150)
        plt.plot(logistic_model.betas_traces[:it + 1, 0], logistic_model.betas_traces[:it + 1, 1], color='black',
                 alpha=.5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(fr'$\beta_0$', fontsize=15)
        plt.ylabel(fr'$\beta_1$', fontsize=15)
        filename = f'logistic_videos/{3 * it + 1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        filenames.append(filename)

        plt.figure(facecolor='white')
        plt.grid(False)
        plt.contourf(beta_0_grid, beta_1_grid, np.array(np.log(posteriors_betas)).reshape(n_points, n_points),
                     cmap='coolwarm', levels=150, alpha=.2)
        plt.plot(logistic_model.betas_traces[:it + 1, 0], logistic_model.betas_traces[:it + 1, 1], color='black',
                 alpha=.5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(fr'$\beta_0$', fontsize=15)
        plt.ylabel(fr'$\beta_1$', fontsize=15)
        plt.scatter(logistic_model.proposals[it, 0], logistic_model.proposals[it, 1], color='black', marker='+', s=150)
        plt.plot([logistic_model.betas_traces[it, 0], logistic_model.proposals[it, 0]],
                 [logistic_model.betas_traces[it, 1], logistic_model.proposals[it, 1]], linestyle='--', color='black',
                 alpha=.5)
        plt.scatter(logistic_model.betas_traces[:it + 1, 0], logistic_model.betas_traces[:it + 1, 1], color='black',
                    marker='+', s=150)
        filename = f'logistic_videos/{3 * it + 2}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        filenames.append(filename)

        plt.figure(facecolor='white')
        plt.grid(False)
        plt.contourf(beta_0_grid, beta_1_grid, np.array(np.log(posteriors_betas)).reshape(n_points, n_points),
                     cmap='coolwarm', levels=150, alpha=.2)
        plt.plot(logistic_model.betas_traces[:it + 1, 0], logistic_model.betas_traces[:it + 1, 1], color='black',
                 alpha=.5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(fr'$\beta_0$', fontsize=15)
        plt.ylabel(fr'$\beta_1$', fontsize=15)
        plt.plot([logistic_model.betas_traces[it, 0], logistic_model.proposals[it, 0]],
                 [logistic_model.betas_traces[it, 1], logistic_model.proposals[it, 1]], linestyle='--', color='black',
                 alpha=.5)
        if logistic_model.accept_array[it]:
            plt.scatter(logistic_model.proposals[it, 0], logistic_model.proposals[it, 1], color='green', marker='+',
                        s=150)
        else:
            plt.scatter(logistic_model.proposals[it, 0], logistic_model.proposals[it, 1], color='red', marker='+',
                        s=150)
        plt.scatter(logistic_model.betas_traces[:it + 1, 0], logistic_model.betas_traces[:it + 1, 1], color='black',
                    marker='+', s=150)
        filename = f'logistic_videos/{3 * it + 3}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        filenames.append(filename)
    save_gif(filenames, gif_name)


def save_gif(filenames, gif_name):
    writer = []
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append(image)
    imageio.mimsave(f'gifs/{gif_name}.gif', writer, fps=1.7)

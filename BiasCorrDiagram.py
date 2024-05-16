import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as fa

def bias_correlation_diagram(modeled_data, observed_data, model_names=None, fig=None, color='pink'):
  """
  Creates a bias-correlation diagram for multiple modeled data sets compared to observed data.

  Args:
      modeled_data (list): List of NumPy arrays containing modeled data from different models.
      observed_data (NumPy array): NumPy array containing the observed data.
      model_names (list, optional): List of strings representing names for each model. Defaults to None.
      fig (matplotlib.figure.Figure, optional): Existing figure to use for plotting. Defaults to None.
      color (str, optional): Color for the scatter plot markers. Defaults to 'pink'.

  Returns:
      matplotlib.figure.Figure: The figure object containing the bias-correlation diagram.
  """

  x_axis_min = -0.5
  x_axis_max = 0.5
  y_axis_max = 1
  y_axis_tick_interval = 0.1

  def calculate_bias(modeled, observed):
    """
    Calculates the normalized mean error (NME) between modeled and observed data.

    Args:
        modeled (NumPy array): Array containing modeled data.
        observed (NumPy array): Array containing observed data.

    Returns:
        float: The NME value.
    """
    return np.mean(modeled) - np.mean(observed)

  def calculate_normalized_bias(modeled, observed):
    """
    Calculates the normalized bias (NMB) between modeled and observed data.

    Args:
        modeled (NumPy array): Array containing modeled data.
        observed (NumPy array): Array containing observed data.

    Returns:
        float: The NMB value.
    """
    return (np.mean(modeled) - np.mean(observed)) / np.std(observed)

  def calculate_correlation(modeled, observed):
    """
    Calculates the correlation coefficient between modeled and observed data.

    Args:
        modeled (NumPy array): Array containing modeled data.
        observed (NumPy array): Array containing observed data.

    Returns:
        float: The correlation coefficient.
    """
    return np.corrcoef(observed, modeled)[0, 1]

  def calculate_rmse(modeled, observed):
    """
    Calculates the root mean squared error (RMSE) between modeled and observed data.

    Args:
        modeled (NumPy array): Array containing modeled data.
        observed (NumPy array): Array containing observed data.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(((modeled - observed)**2).mean())

  # Calculate bias and correlation for each model
  biases = np.array([calculate_bias(m, observed_data) for m in modeled_data])
  correlations = np.array([calculate_correlation(m, observed_data) for m in modeled_data])

  # Limit bias and correlation values to specified ranges
  biases = np.clip(biases, x_axis_min, x_axis_max)
  correlations = np.clip(correlations, 0.0, y_axis_max)

  # Configure matplotlib for background processing and plot style
  plt.ioff()
  mpl.use('Agg')
  mpl.pyplot.style.use('ggplot')

  # Create figure if not provided
  if not fig:
    fig = plt.figure()
  ax = fig.add_subplot()

  # Plot reference lines for perfect bias and correlation
  ax.scatter([0], [1], c='darkgray', s=100, edgecolor='black')
  ax.plot([0, 0], [0, 1], c='black', lw=1.5)
  ax.plot([-0.5, 0.5], [1, 1], c='black', lw=1.5)

  # Scatter plot for bias and correlation of each model
  ax.scatter(biases, correlations, c=color, s=60, edgecolor='black')

  # Set axis labels and limits
  ax.set_xlabel('Bias (NME)')
  ax.set_xlim([x_axis_min - 0.005, x_axis_max + 0.005])
  ax.set_xticks(np.arange(x_axis_min, x_axis_max + y_axis_tick_interval, y_axis_tick_interval))

  ax.set_ylabel('Correlation')
  ax.set_ylim([0, y_axis_max + 0.005])
  ax.set_yticks(np.arange(0, y_axis_max + y_axis_tick_interval, y_axis_tick_interval))

  # Add model names as annotations if provided
  if model_names:
    for i, txt in enumerate(model_names):
      ax.annotate(txt, (biases[i], correlations[i]), textcoords="offset points", ha='center', xytext=(0, 7))

  return fig

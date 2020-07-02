import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd


def plot_acc_dataprop(prop, acc, method):
    """
    plot the accuracy vs data proportion being used, graph
    credits to: Alex Olteanu (https://www.dataquest.io/blog/making-538-plots/) for the plot style
    :return: None
    """

    style.use('fivethirtyeight')

    df = pd.DataFrame({'prop': prop, 'acc': acc, 'method': method})
    # Colorblind-friendly colors
    colors = [[0, 0, 0, 1], [230 / 255, 159 / 255, 0, 1], [86 / 255, 180 / 255, 233 / 255, 1], [0, 158 / 255, 115 / 255, 1],
              [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1]]

    fig, ax = plt.subplots()

    # The previous code we modify
    for i, (key, grp) in enumerate(df.groupby('method')):
        fte_graph = grp.plot(x='prop', y='acc', figsize=(14, 12), color=[colors[i]], legend=False, ax=ax)

        # The previous code that remains unchanged
        fte_graph.tick_params(axis='both', which='major', labelsize=22)
        fte_graph.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        fte_graph.set_yticklabels(labels=['0   ', '10   ', '20   ', '30   ', '40   ', '50   ', '60   ', '70   ',
                                          '80   ', '90   ', '100%'])
        fte_graph.axhline(y=-0.01, color='black', linewidth=1.3, alpha=.7)
        fte_graph.set_xlabel("Labeled percentage of dataset (%)")
        fte_graph.set_ylabel("Accuracy")
        # fte_graph.xaxis.label.set_visible(False)
        fte_graph.set_xlim(left=1, right=35)
        fte_graph.text(x=-1, y=120, s="A comparison of semi-supervised methods",
                       fontsize=26, weight='bold', alpha=.75)
        fte_graph.text(x=-1, y=110,
                       s='Plotting the accuracy as a function of the percentage of dataset being considered as\n'
                         'unlabeled data',
                       fontsize=19, alpha=.85)
        fte_graph.text(x=15, y=97, s='FixMatch', color=colors[0], weight='bold', rotation=2,
                       backgroundcolor='#f0f0f0')
    plt.grid(linestyle='--')
    plt.show()


if __name__ == "__main__":
    plot_acc_dataprop(
        [3.5, 7.1, 14.28, 28.57],
        [91.86, 93.44, 94.80, 95.92],
        [0, 0, 0, 0]
    )

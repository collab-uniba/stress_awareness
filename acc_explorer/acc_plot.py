
import constant
import plotly.graph_objs as go


def plot_axes(data):
    # TODO Sostituire num_rows e index con l'orario
    names = list(data.keys())
    metrics_data = list(data.values())
    num_values = metrics_data[0].shape[0]

     # Switch from hz to seconds
    index = [v/constant.NUM_HZ for v in range(num_values)]

    fig = go.Figure()

    for metric in names:
        fig.add_trace(go.Scatter(x=index, y=data[metric],
                                 mode='lines',
                                 name=metric,
                                 visible='legendonly'))
    fig.update_layout(xaxis_title='Seconds', yaxis_title='axes')
    fig.show()

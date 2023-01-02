import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch_geometric.utils as U
from torch.autograd import Variable


def train_loop(model, dataset, optimizer, loss_fn):
    model.train()
    correct = 0
    total_num = 0
    for graph in dataset:
        optimizer.zero_grad()
        edge_index = graph.edge_index
        nodes = model(graph.x, edge_index)
        pred = nodes
        correct += (pred.argmax(dim=-1) == graph.y[0]).sum().item()
        loss = loss_fn(pred, graph.y[0])
        loss.backward()
        optimizer.step()
        total_num += 1
    return loss, correct / total_num


def test_loop(model, dataset):
    model.eval()
    correct = 0
    total_num = 0
    with torch.no_grad():
        for graph in dataset:
            edge_index = graph.edge_index
            nodes = model(graph.x, edge_index)
            pred = nodes
            correct += (pred.argmax(dim=-1) == graph.y[0]).sum().item()
            total_num += 1
    return correct / total_num


def train(model, dataset, num_epochs, loss_fn=nn.CrossEntropyLoss(), optimizer_cls=torch.optim.AdamW, lr=1e-4, weight_decay=0):
    train_acc = []
    test_acc = []

    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        loss, correct_train = train_loop(model, dataset, optimizer, loss_fn)
        correct_test = test_loop(model, dataset)
        train_acc.append(correct_train)
        test_acc.append(correct_test)
        if epoch % 10 == 0:
            print("epoch={}, loss={}, accuracy={}".format(epoch, loss.item(), correct_test))

    return {'train_acc': train_acc, 'test_acc': test_acc}


def plot_graph(data, grads, labels=None, rows=2, pos=None):
    fig, ax = plt.subplots(rows, grads.shape[1] // rows + 1, figsize=(16, 6))
    ax = ax.flatten()
    for a in ax:
        a.set_visible(False)
    norm = max(abs(grads.min()), grads.max())
    graph = U.to_networkx(data, to_undirected=True)
    if pos is None:
        pos = nx.spring_layout(graph)
    for i in range(grads.shape[1]):
        g_norm = grads[:, i] / norm
        colors = [(1, 1 - f, 1 - f) if f > 0 else (1 + f, 1 + f, 1) for f in g_norm]
        ax[i].set_visible(True)
        if labels is not None:
            ax[i].set_title(labels[i])
        nx.draw(graph, node_color=colors, ax=ax[i], node_size=50, pos=pos)
    return pos


def get_grad(data, model):
    model.zero_grad()
    x = Variable(data.x, requires_grad=True)
    x.retain_grad()
    y = model(x, data.edge_index)
    y[data.y].backward(torch.ones_like(y[data.y]))
    return x.grad

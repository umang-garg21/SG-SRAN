# -*- coding: utf-8 -*-
# Auto-generated from simple_tasks_and_symmetry.ipynb
# %%
# %% [markdown]
# # Simple Tasks and Symmetry
# ### using the `e3nn` repository
# ### tutorial by: Tess E. Smidt (`blondegeek`)

# %% [markdown]
# ### There are some unintuitive consequences of using E(3) equivariant neural networks. 
# One example is that the symmetry your output has to be equal to or higher than the symmetry of your input. The following 3 simple tasks are to help demonstrate this:
# * Task 1: Distort a rectangle to a square.
# * Task 2: Distort a square to a rectangle.
# * Task 3: Distort a square to a rectangle -- with symmetry breaking (using representation theory).
# * Task 4: Distort a square to a rectangle -- with symmetry breaking (using gradients to change input).
#
# We will see that we can quickly do Task 1, but not Task 2. Only by using symmetry breaking in Task 3 and Task 4 are we able to distort a square into a rectangle.

# %%
import torch
from functools import partial

from e3nn import o3 
from e3nn.kernel_mod import Kernel
from e3nn.tensor.spherical_tensor import SphericalTensor

import plotly
import plotly.graph_objects as go

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# %%
# Define out geometry
square = torch.tensor(
    [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]
)
square -= square.mean(-2)
sx, sy = 0.5, 1.5
rectangle = square * torch.tensor([sx, sy, 0.])
rectangle -= rectangle.mean(-2)

N, _ = square.shape

markersize = 15

def plot_task(ax, start, finish, title, marker=None):
    ax.plot(torch.cat([start[:, 0], start[:, 0]]), 
            torch.cat([start[:, 1], start[:, 1]]), 'o-', 
            markersize=markersize + 5 if marker else markersize, 
            marker=marker if marker else 'o')
    ax.plot(torch.cat([finish[:, 0], finish[:, 0]]), 
            torch.cat([finish[:, 1], finish[:, 1]]), 'o-', markersize=markersize)
    for i in range(N):
        ax.arrow(start[i, 0], start[i, 1], 
                 finish[i, 0] - start[i, 0], 
                 finish[i, 1] - start[i, 1],
                 length_includes_head=True, head_width=0.05, facecolor="black", zorder=100)

    ax.set_title(title)
    ax.set_axis_off()

fig, axes = plt.subplots(1, 2, figsize=(9, 6))
plot_task(axes[0], rectangle, square, "Task 1: Rectangle to Square")
plot_task(axes[1], square, rectangle, "Task 2: Square to Rectangle")


# %% [markdown]
# In these tasks, we want to move 4 points in one configuration to another configuration. The input to the network will be the initial geometry and features on that geometry. The output will be used to signify "displacement" of each point to the new configuration. We can represent displacement in a couple different ways. The simplest way is to represent a displacement as an L=1 vector, `Rs=[(1, 1]]`. However, to better illustrate the symmetry properties of the network, we instead are going to use a spherical harmonic signal or more specifically, the peak of the spherical harmonic signal, to signify the displacement of the original point.
#
# First, we set up a very basic network that has the same representation list `Rs = [(1, L) for L in range(5 + 1)]` throughout the entire network. The input will be a spherical tensor with representation `Rs` and the output will also be a spherical tensor with representation `Rs`. We will interpret the output of the network as a spherical harmonic signal where the peak location will signify the desired displacement.

# %% [markdown]
# ## For these examples, we will used the default `e3nn.networks.GatedConvNetwork` class for our model

# %%
from e3nn.networks import GatedConvNetwork
L_max = 5
Rs = [(1, L) for L in range(L_max + 1)]
Network = partial(GatedConvNetwork,
                  Rs_in=Rs,
                  Rs_hidden=Rs,
                  Rs_out=Rs,
                  lmax=L_max,
                  max_radius=3.0,
                  kernel=Kernel)


# %%
from e3nn.networks import GatedConvParityNetwork
L_max = 5
Rs = [(1, L, (-1)**L) for L in range(L_max + 1)]
Network = partial(GatedConvParityNetwork, 
                  Rs_in=Rs, 
                  mul=5, 
                  Rs_out=Rs, 
                  lmax=L_max, 
                  max_radius=3.0, 
                  layers=1)


# %% [markdown]
# ## Task 1: Distort a rectangle to square.
# In this task, our input is a four points in the shape of a rectangle with simple scalars (1.0) at each point. The task is to learn to displace the points to form a (more symmetric) square.

# %%
model = Network()

optimizer = torch.optim.Adam(model.parameters(), 1e-2)


# %%
input = torch.zeros((L_max + 1)**2)
input[0] = 1

displacements = square - rectangle
projections = torch.stack([
    SphericalTensor.from_geometry(r, L_max).signal 
    for r in displacements
])


# %%
for i in range(51):
    output = model(input.repeat(1, 4, 1), rectangle[None])[0]
    loss = (output - projections).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, loss)


# %%
def plot_output(start, finish, features, start_label, finish_label, bound=None):
    if bound is None:
        bound = max(start.norm(dim=1).max(), finish.norm(dim=1).max()).item()
    axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title='',
        nticks=3,
        range=[-bound, bound]
    )

    resolution = 500
    layout = dict(
        width=resolution,
        height=resolution,
        scene=dict(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=2),
                projection=dict(type='perspective'),
            ),
        ),
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )

    traces = [
        go.Scatter3d(x=start[:, 0], y=start[:, 1], z=start[:, 2], mode="markers", name=start_label),
        go.Scatter3d(x=finish[:, 0], y=finish[:, 1], z=finish[:, 2], mode="markers", name=finish_label),
    ]
    
    for center, signal in zip(start, features):
        r, f = SphericalTensor(signal.detach()).plot(center=center)
        traces += [go.Surface(x=r[..., 0], y=r[..., 1], z=r[..., 2], surfacecolor=f.numpy(), showscale=False)]
        
    return go.Figure(traces, layout=layout)


# %%
output = model(input.repeat(1, 4, 1), rectangle[None])[0]
fig = plot_output(rectangle, square, output, "Rectangle", "Square")
fig.show()


# %% [markdown]
# ### And let's check that it's equivariant

# %%
angles = o3.rand_angles()
rot = -o3.rot(*angles)  # rotation + parity

rot_rectangle = torch.einsum('xy,jy->jx', rot, rectangle)
rot_square = torch.einsum('xy,jy->jx', rot, square)

output = model(input.repeat(1, 4, 1), rot_rectangle[None])[0]
fig = plot_output(rot_rectangle, rot_square, output, "Rectangle", "Square")
fig.show()


# %% [markdown]
# ## Task 2: Now the reverse! Distort a square to rectangle.
# In this task, our input is a four points in the shape of a square with simple scalars (1.0) at each point. The task is to learn to displace the points to form a (less symmetric) rectangle. Can the network learn this task?

# %%
model = Network()

optimizer = torch.optim.Adam(model.parameters(), 1e-2)


# %%
input = torch.zeros((L_max + 1)**2)
input[0] = 1

displacements = rectangle - square
projections = torch.stack([
    SphericalTensor.from_geometry(r, L_max).signal 
    for r in displacements
])


# %%
for i in range(51):
    output = model(input.repeat(1, 4, 1), square[None])[0]
    loss = (output - projections).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, loss)


# %% [markdown]
# ## Hmm... seems to get stuck. Let's try more iterations.

# %%
for i in range(51):
    output = model(input.repeat(1, 4, 1), square[None])[0]
    loss = (output - projections).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, loss)


# %% [markdown]
# ## It's stuck. What's going on?

# %%
fig = plot_output(square, rectangle, output, "Square", "Rectangle")
fig.show()


# %% [markdown]
# ### The symmetry of the output must be higher or equal to the symmetry of the input! 
# To be able to do this task, you need to give the network more information -- information that breaks the symmetry to that of the desired output. The square has a point group of $D_{4h}$ (16 elements) while the rectangle has a point group of $D_{2h}$ (8 elements).
#
# #### A technical note (for those who are interested).
# In this example, if we do not use a network equivariant to [parity](https://en.wikipedia.org/wiki/Parity_(physics)) (e.g. using `GatedConvNetwork` instead of `GatedConvParityNetwork`) -- we would be only sensitive to the fact that the square has $C_4$ symmetry while the rectangle has $C_2$ symmetry.

# %% [markdown]
# ## Task 3: Fixing Task 2. Distort a square into a rectangle -- now, with symmetry breaking (using representation theory)!
#
# In this task, our input is four points in the shape of a square with simple scalars (1.0) AND a contribution for the $x^2 - y^2$ feature at each point. The task is to learn to displace the points to form a (less symmetric) rectangle. Can the network learn this task?

# %%
model = Network()

optimizer = torch.optim.Adam(model.parameters(), 1e-2)


# %%
input = torch.zeros((L_max + 1)**2)

# Breaking x and y symmetry with x^2 - y^2 component
input[0] = 1
input[8] = 1  # x^2 - y^2

displacements = rectangle - square
projections = torch.stack([
    SphericalTensor.from_geometry(r, L_max).signal 
    for r in displacements
])


# %%
for i in range(51):
    output = model(input.repeat(1, 4, 1), square[None])[0]
    loss = (output - projections).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, loss)


# %%
fig = plot_output(square, rectangle, output, "Square", "Rectangle")
fig.show()


# %% [markdown]
# ### What is $x^2 - y^2$ the term doing? It's breaking the symmetry along the $\hat{x}$ and $\hat{y}$ directions.
# Notice how the shape below is an ellisoid elongated in the y direction and squished in the x. This isn't the only pertubation we could've added, but it is the most symmetric.

# %%
fig = plot_output(square, square, 0.3 * input.repeat(4, 1), '', '', bound=0.75)
fig.show()


# %% [markdown]
# ### Sure, but where did the $x^2 - y^2$ come from?
# It's a bit of a complicated story, but at the surface level here it is: [Character tables](https://en.wikipedia.org/wiki/Character_table) are handy tabulations of how certain spherical tensor datatypes transform under that group symmetry. The rows are irreducible representations (irrep for short) and the columns are similar elements of the group (called [conjugacy classes](https://en.wikipedia.org/wiki/Conjugacy_class)). Character tables are most commonly seen for finite groups of $E(3)$ symmetry as they are used extensively in solid state physics, crystallography, chemistry, etc. Next to the part of the table with the "characters", there are often columns showing linear, quadratic, and cubic functions (meaning they are of order 1, 2, and 3) that transform in the same way as a given irrep.
#
# So, a square has a point group symmetry of $D_{4h}$ while a rectangle has a point group symmetry of $D_{2h}$
#
# If we look at column headers of character tables for $D_{4h}$ and $D_{2h}$...
# * [$D_{4h}$ Character Table](http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=604&option=4)
# * [$D_{2h}$ Character Table](http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=602&option=4)
#
# ... we can see that the irrep $B_{1g}$ of $D_{4h}$ that has -1's in the columns for all the symmetry operations that $D_{2h}$ DOESN'T have and if we look down that row to the column "quadratic functions" we see, voila $x^2 - y^2$. So, to break all those symmetries that $D_{4h}$ has that $D_{2h}$ DOESN'T have -- we add a non-zero contribution to the $x^2 - y^2$ component of our spherical harmonic tensors.
#
# #### WARNING: Character tables are written down with specific coordinate system conventions. For example, the $\hat{z}$ axis always points along the highest symmetry axis, $\hat{y}$ along the next highest, etc. We have specifically set up our problem have a coordinate frame that matches these conventions.
#
# #### A technical note (for those who are interested).
# Again, in this example if we leave out parity (by using `GatedConvNetwork` instead of `GatedParityConvNetwork`), we would only be sensitive to the fact that the square has $C_4$ symmetry while the rectangle has $C_2$ symmetry. However, you can check the character tables for the point groups [$C_4$](http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=204&option=4) and [$C_2$](http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=202&option=4) to see that the arguement above still holds for the $x^2 - y^2$ order parameter.

# %% [markdown]
# ## Task 4: Fixing Task 2 without having to read character tables like Task 4. Distort a square into a rectangle -- now, with symmetry breaking (using gradients to change the input)!
#
# In this task, our input is four points in the shape of a square with simple scalars (1.0) AND then we LEARN how to change the inputs to break symmetry such that we can fit a better model.

# %%
model = Network()

input = torch.zeros((L_max + 1)**2, requires_grad=True)
with torch.no_grad():
    input[0] = 1

displacements = rectangle - square
projections = torch.stack([
    SphericalTensor.from_geometry(r, L_max).signal 
    for r in displacements
])

param_optimizer = torch.optim.Adam(model.parameters(), 1e-2)
input_optimizer = torch.optim.Adam([input], 1e-3)


# %% [markdown]
# ## First, we'll train the model until it gets stuck.

# %%
for i in range(21):
    output = model(input.repeat(4, 1)[None], square[None])[0]
    loss = (output - projections).pow(2).mean()
    param_optimizer.zero_grad()
    loss.backward()
    param_optimizer.step()
    if i % 10 == 0:
        print(i, loss)


# %% [markdown]
# ## This gets stuck like before. So let's try alternating between updating our input and updating the model.

# %%
for i in range(101):
    output = model(input.repeat(4, 1)[None], square[None])[0]
    loss = (output - projections).pow(2).mean()
    param_optimizer.zero_grad()
    loss.backward()
    param_optimizer.step()
    if i % 10 == 0:
        print(i, 'model loss: ', loss)


    output = model(input.repeat(4, 1)[None], square[None])[0]
    loss = (output - projections).pow(2).mean()
    # Add sparse penalty to L=2
    loss += 1e-3 * (input[1:].abs()).mean()
    input_optimizer.zero_grad()
    loss.backward()
    input_optimizer.step()
    
    # only allow L=2 to evolve
    with torch.no_grad():
        input[0] = 1  # L=0
#         input[1**2:2**2] = 0  # L=1
#         input[3**2:] = 0  # L>=3
        
    if i % 10 == 0:
        print(i, 'input loss: ', loss)


# %% [markdown]
# ### If we examine the input, we should see that the only components that are (largely) non-zero are the scalar features (which are all 1's) and the features that transform as the $B_{1g}$ irrep of $D_{4h}$ such as the L=2 feature corresponding to $x^2 - y^2$, which is the 5th element of the L=2 array, and L=4 the feature corresponding to $(x^2-y^2)(7z^2 - r^2)$, which is the 7th element of the L=4 array.

# %%
for L in range(L_max + 1):
    print("L={}".format(L))
    print(input[L**2:(L+1)**2].detach().numpy().round(3))


# %% [markdown]
# ## This plot shows what the new input looks like. It's similar to the above plot from Task 3.

# %%
fig = plot_output(square, square, input.repeat(4, 1), '', '', bound=1)
fig.show()


# %%


import sympy as sp
import veclib.spacetime as spacetime
import itertools

class Tensor: 
    def __init__(self, name, components, indices = []): 
        """
        Initializes a Tensor object.
        
        Args:
            components (list or MutableDenseNDimArray): The components of the tensor.
            indices (list of int): A list indicating the index types:
                                   -1 for lower (covariant), 1 for upper (contravariant).
            name (str): The symbol representing the tensor
        """

        if isinstance(components, sp.Basic):
            self.components = components
            self.rank = 0
            self.indices = indices
        elif isinstance(components, sp.MutableDenseNDimArray):
            self.components = sp.MutableDenseNDimArray(components)
            self.rank = self.components.rank()
            self.indices = indices
        elif isinstance(components, sp.matrices.dense.MutableDenseMatrix):
            self.components = sp.MutableDenseNDimArray(components.tolist())
            self.rank = self.components.rank()
            self.indices = indices
        elif isinstance(components, Tensor):
            self.components = sp.MutableDenseNDimArray(components.components)
            self.rank = components.rank
            self.indices = components.indices
            print(self.rank, self.indices)
        else: 
            self.components = sp.MutableDenseNDimArray(components)
            self.rank = self.components.rank()
            self.indices = indices

        self.name = name
        
        if len(self.indices) != self.rank: 
            raise ValueError(
                    f"Amount of indices {len(indices)} must match the tensor rank {self.rank}."
                )
        if len(indices) != 0:
            if not all(i in (-1,1) for i in indices):
                raise ValueError("Indices must be a list of -1 (lower) or 1 (upper).")


    def __repr__(self):
        """
        Machine-readable representation of the Tensor.
        """
        return f"Tensor(components={self.components}, indices={self.indices}, name={self.name})"

    def _repr_latex_(self):

        # LaTeX representation for Jupyter notebooks
        index_str = "".join(
            [
                f"\\!\\,_{{{chr(945 + i)}}}" if idx == -1 else f"\\!\\,^{{{chr(945 + i)}}}"
                for i, idx in enumerate(self.indices)
            ]
        )
        
        latex_components = sp.latex(self.components)
        if len(self.name) > 1:
            return f"$({self.name}){index_str} = {latex_components}$"
        else:
            return f"${self.name}{index_str} = {latex_components}$"

    def __str__(self):
        """
        Human-readable representation of the Tensor.
        """
        index_str = "".join("'" if i == 1 else "," for i in self.indices)
        return f"{self.name}{index_str}:\n{self.components}"

    def shortStr(self):
        """
        Human-readable representation of the Tensor.
        """
        index_str = "".join("'" if i == 1 else "," for i in self.indices)
        return f"{self.name}{index_str}"

    def get_components(self):
        return self.components

    def raise_index(self, index_pos = 0):
        """
        Raises the index at index_pos from lower to upper using the global metric.
        
        Args:
            index_pos (int): The position of the index to be raised (from lower to upper).
        """
        if spacetime.metric == None:
            raise ValueError("Global metric is not set.")

        metric_inv = spacetime.metric_inv

        if not index_pos in range(self.rank):
            raise ValueError(f"index_pos {index_pos} exceeds tensor rank {self.rank}")

        if self.indices[index_pos] == -1:
            new_components = sp.zeros(*self.components.shape)
            new_indices = self.indices.copy()
            new_indices[index_pos] = 1

            #contraction with the metric
            if self.rank > 1:
                new_components = sp.zeros(*self.components.shape)
                new_indices = self.indices.copy()
                new_indices[index_pos] = 1
                for i in itertools.product(range(spacetime.dim), repeat = self.rank):
                    j = list(i)
                    for a in range(spacetime.dim):
                        j[index_pos] = a
                        new_components[i] += (self.components[j] * metric_inv[a,i[index_pos]]).simplify()
            else:
                new_components = sp.MutableDenseNDimArray([0] * spacetime.dim)
                new_indices = [1]
                for i in range(spacetime.dim):
                    for a in range(spacetime.dim):
                        new_components[i] += (self.components[a] * metric_inv[a,i]).simplify()
            return Tensor(self.name, new_components, new_indices)
        else:
            raise ValueError(f"Cannot raise index at position {index_pos} of {self.shortStr()}: already upper.")

    def lower_index(self, index_pos = 0):
        """
        Raises the index at index_pos from lower to upper using the global metric.
        
        Args:
            index_pos (int): The position of the index to be raised (from lower to upper).
        """
        if spacetime.metric == None:
            raise ValueError("Global metric is not set.")

        metric = spacetime.metric

        if not index_pos in range(self.rank):
            raise ValueError(f"Index position {index_pos} exceeds tensor rank {self.rank}")

        if self.indices[index_pos] == 1:
            #contraction with the metric
            print(f"self.rank = {self.rank}")
            if self.rank > 1:
                new_components = sp.zeros(*self.components.shape)
                new_indices = self.indices.copy()
                new_indices[index_pos] = -1
                for i in itertools.product(range(spacetime.dim), repeat = self.rank):
                    j = list(i)
                    for a in range(spacetime.dim):
                        j[index_pos] = a
                        new_components[i] += (self.components[j] * metric[a,i[index_pos]]).simplify()
            else:
                new_components = sp.MutableDenseNDimArray([0] * spacetime.dim)
                new_indices = [-1]
                for i in range(spacetime.dim):
                    for a in range(spacetime.dim):
                        new_components[i] += (self.components[a] * metric[a,i]).simplify()
            return Tensor(self.name, new_components, new_indices)
        else:
            raise ValueError(f"Cannot lower index at position {index_pos} of {self.shortStr()}: already lower.")

    def tensor_product(self, other_tensor):
        """
        Computes the tensor product of the current tensor with another tensor.

        The tensor product combines the components of the two tensors into a higher-rank tensor,
        concatenates their indices, and formats the name to reflect the operation.

        Args:
            other_tensor (Tensor): The tensor to compute the tensor product with.

        Returns:
            Tensor: A new tensor representing the tensor product.
        """
        # Perform the tensor product of components and store the result as a mutable dense array
        product_components = sp.MutableDenseNDimArray(sp.tensorproduct(self.components, other_tensor.components))
        
        # Construct the name for the resulting tensor
        product_name = self.name + "⊗" + other_tensor.name
        
        # Combine the indices of both tensors
        product_indices = self.indices + other_tensor.indices

        # Return a new Tensor object representing the tensor product
        return Tensor(product_name, product_components, product_indices)


    def swap_indices(self, index1 = 0, index2 = 1): 
        # Check if the indices refer to the same type (upper or lower)
        if self.indices[index1] != self.indices[index2]:
            raise ValueError("Cannot exchange upper with lower index")
        
        # Ensure the indices are within the tensor rank
        if index1 >= self.rank or index2 >= self.rank:
            raise ValueError(f"One of indices ({index1}, {index2}) exceeds tensor rank {self.rank}")
        
        # Swap the indices
        new_indices = self.indices.copy()
        new_indices[index1], new_indices[index2] = new_indices[index2], new_indices[index1]

        # Create an order list to reorder the axes and apply transpose
        order = list(range(self.components.rank()))
        order[index1], order[index2] = order[index2], order[index1]  # Swap the axes

        # Apply the transpose to the components based on the new axis order
        new_components = sp.permutedims(self.components, order)

        # Return the new Tensor with swapped indices and components
        if len(self.name.replace("∂", "")) == 1:
            return Tensor(self.name, new_components, new_indices)    
        return Tensor("∎", new_components, new_indices)

    def partial_gradient(self):
        """
        returns the partial derivative gradient diff_a T of the tensor. 
        NOTE: The index a of the partial derivative is the FIRST index of the new tensor.
        """
        if self.rank > 0:
            result = sp.MutableDenseNDimArray.zeros(spacetime.dim, *self.components.shape)
        elif self.rank == 0:
            result = sp.MutableDenseNDimArray.zeros(spacetime.dim)
        else: 
            raise ValueError(f"Tensor {self.name} has invalid rank {self.rank}.")

        if self.rank > 0:
            for i in itertools.product(range(spacetime.dim), repeat = result.rank()):
                entry =  sp.diff(self.components[i[1:]], spacetime.coords[i[0]]).simplify()
                result[i] = entry
        else:
            for i in range(spacetime.dim):
                result[i] = sp.diff(self.components, spacetime.coords[i]).simplify()
        return Tensor(f"∂{self.name}", result, [-1] + self.indices)

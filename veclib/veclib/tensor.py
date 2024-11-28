import sympy as sp
import veclib.spacetime as spacetime
import veclib.combinatorics as combinatorics
import itertools
from IPython.display import display, Math

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
            return f"$[{self.name}]{index_str} = {latex_components}$"
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

    def show_component(self, indices=[]):
        from sympy.printing.latex import greek_letters_set  # To identify Greek letters

        if len(indices) != self.rank:
            raise ValueError(f"Invalid index count {len(indices)} for rank {self.rank} tensor.")
        
        def ensure_italic(symbol):
            """Ensure all symbols are italicised in LaTeX output."""
            # If it's a Greek letter, let SymPy handle it as usual
            if str(symbol) in greek_letters_set:
                return sp.latex(sp.Symbol(symbol))
            # Otherwise, wrap in \mathit{} to enforce italics
            return f"\\mathit{{{symbol}}}"

        bracketed_name = ensure_italic(self.name)
        
        #sacred conditional, do not change (took me 20mins to fix)
        if len(self.name) > 1 and bracketed_name.count("\\") != 2:
            bracketed_name = "[" + bracketed_name + "]"

        if self.rank == 0:
            display(Math(bracketed_name + " = " + sp.latex(self.components)))
        else:
            component_string = bracketed_name

            for n, i in enumerate(indices):
                if self.indices[n] == -1:  # Covariant index
                    component_string += f"\\!\\,_{{{sp.latex(spacetime.coords[i])}}}"
                else:  # Contravariant index
                    component_string += f"\\!\\,^{{{sp.latex(spacetime.coords[i])}}}"
            
            # Render the LaTeX string
            display(Math(component_string + " = " + sp.latex(self.components[indices])))

    def show_components(self):
        if self.rank == 0:
            self.show_component()
        else:
            printed_components = 0
            for i in itertools.product(range(spacetime.dim), repeat = self.rank):
                if self.components[i] != 0:
                    self.show_component([*i])
                    printed_components += 1
            if printed_components == 0: #if tensor is zero, write this down briefly
                index_str = "".join(
                    [
                        f"\\!\\,_{{{chr(945 + i)}}}" if idx == -1 else f"\\!\\,^{{{chr(945 + i)}}}"
                        for i, idx in enumerate(self.indices)
                    ]
                )
                display(Math(self.name + index_str + "= 0"))
                



    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Can only add another Tensor")
        if self.indices != other.indices:
            raise ValueError(f"Cannot add tensors with different index types {self.indices}, {other.indices}")
        if self.components.shape != other.components.shape:
            raise ValueError("Cannot add tensors with different shapes")

        # Perform component-wise addition
        new_components = self.components + other.components

        return Tensor(f"({self.name} + {other.name})", new_components, self.indices)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Can only subtract another Tensor")
        if self.indices != other.indices:
            raise ValueError(f"Cannot subtract tensors with different index types {self.indices}, {other.indices}")
        if self.components.shape != other.components.shape:
            raise ValueError("Cannot subtract tensors with different shapes")

        # Perform component-wise subtraction
        new_components = self.components - other.components

        return Tensor(f"({self.name} - {other.name})", new_components, self.indices)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            # Tensor product
            new_components = sp.MutableDenseNDimArray(
                sp.tensorproduct(self.components, other.components)
            )
            new_indices = self.indices + other.indices
            product_symbol = ""
            if not (other.rank == 0 or self.rank == 0):
                product_symbol = "⊗"
            return Tensor(f"{self.name}{product_symbol}{other.name}", new_components, new_indices)
        elif isinstance(other, (int, float, sp.Basic)):
            # Element-wise scalar multiplication
            new_components = self.components * other
            return Tensor(f"({other}{self.name})", new_components, self.indices)
        else:
            raise TypeError(f"Cannot multiply Tensor with type {type(other)}")

    def __rmul__(self, other):
        # For scalar * tensor (reverse multiplication)
        return self.__mul__(other)

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

            #contraction with the metric
            if self.rank > 1:
                new_components = sp.MutableDenseNDimArray.zeros(*self.components.shape)
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
            if self.rank > 1:
                new_components = sp.MutableDenseNDimArray.zeros(*self.components.shape)
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

    def contract(self, index1 = 0, index2 = 1):
        if not index1 in range(self.rank) or not index2 in range(self.rank):
            raise ValueError(f"Indices {index1, index2} exceed tensor rank {self.rank}")

        contractable_tensor = Tensor(self.name, self)

        if self.indices[index1] == self.indices[index2]:
            if self.indices[index1] == -1:
                contractable_tensor = contractable_tensor.raise_index(index1)
            else:
                contractable_tensor = contractable_tensor.lower_index(index1)

        contracted_indices = [x for idx, x in enumerate(self.indices) if idx != index1 and idx != index2]
        contracted_components = sp.tensorcontraction(contractable_tensor.components, [index1, index2])

        result = Tensor(self.name, contracted_components, contracted_indices)
        result = result.trigsimp()
        result = result.simplify()
        return result



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
        if len(self.name.replace("∂", "").replace("∇", "")) == 1:
            return Tensor(self.name, new_components, new_indices)    
        return Tensor("...", new_components, new_indices)

    def reorder_indices(self, new_order):
        #input check
        if len(new_order) != self.rank:
            raise ValueError(f"Incorrect amount of index positions specified ({self.rank} needed, {len(new_order)} given).")
        if not all(i in range(self.rank) for i in new_order) or len(new_order) != len(set(new_order)):   
            raise ValueError(f"Invalid order {new_order}")

        new_components = sp.permutedims(self.components, new_order)
        new_indices = [self.indices[i] for i in new_order]

        return Tensor("...", new_components, new_indices)

    def symmetrise_pair(self, index1, index2, normalise = True):
        if self.rank < 2:
            raise ValueError("Cannot symmetrise tensor of rank lower than 2.")
        if not all(idx in [index1, index2] in range(self.rank)):
            raise ValueError(f"Invalid indices [{index1}, {index2}] to symmetrise.")
        if index1 == index2:
            raise ValueError(f"Indices to symmetrise [{index1}, {index2}], cannot be equal.")
        if normalise:
            return sp.Rational(1,2) * (self + self.swap_indices(index1, index2))
        (self + self.swap_indices(index1, index2))

    def symmetrise(self, index_positions = None, normalise = True):
        if index_positions == None:
            index_positions = range(self.rank)
        if abs(sum(self.indices[i] for i in index_positions)) != len(index_positions):
            raise ValueError("Cannot permute symmetrise a mix of upper and lower indices.")
        result = Tensor(self.name, sp.MutableDenseNDimArray.zeros(*self.components.shape), self.indices)
        permuted_indices = combinatorics.permute(range(self.rank), index_positions)
        for perm in permuted_indices:
            result += self.reorder_indices(perm)

        if normalise:
            result *= sp.Rational(1,sp.factorial(len(index_positions)))
        result.name = self.name
        result = result.trigsimp().simplify()
        return result

    def antisymmetrise_pair(self, index1, index2, normalise = True):
        if self.rank < 2:
            raise ValueError("Cannot antisymmetrise tensor of rank lower than 2.")
        if not all(idx in range(self.rank) for idx in [index1, index2]):
            raise ValueError(f"Invalid indices [{index1}, {index2}] to antisymmetrise.")
        if index1 == index2:
            raise ValueError(f"Indices to antisymmetrise [{index1}, {index2}], cannot be equal.")
        if normalise:
            return sp.Rational(1, 2) * (self - self.swap_indices(index1, index2))
        return (self - self.swap_indices(index1, index2))

    def antisymmetrise(self, index_positions = None, normalise = True):
        if index_positions == None:
            index_positions = range(self.rank)
        if abs(sum(self.indices[i] for i in index_positions)) != len(index_positions):
            raise ValueError("Cannot permute symmetrise a mix of upper and lower indices.")
        result = Tensor(self.name, sp.MutableDenseNDimArray.zeros(*self.components.shape), self.indices)
        permuted_indices = combinatorics.permute(range(self.rank), index_positions)
        for perm in permuted_indices:
            result += combinatorics.permutation_sign(perm) * self.reorder_indices(perm)

        if normalise:
            result *= sp.Rational(1,sp.factorial(len(index_positions)))
        result.name = self.name
        result = result.trigsimp().simplify()
        return result

    def simplify(self):
        if self.rank > 0:
            return Tensor(self.name, self.components.applyfunc(lambda expr: expr.simplify()), self.indices)
        else:
            return Tensor(self.name, self.components.simplify())

    def cancel(self):
        if self.rank > 0:
            return Tensor(self.name, self.components.applyfunc(lambda expr: expr.cancel()), self.indices)
        else:
            return Tensor(self.name, self.components.cancel())

    def expand(self):
        if self.rank > 0:
            return Tensor(self.name, self.components.applyfunc(lambda expr: expr.expand()), self.indices)
        else:
            return Tensor(self.name, self.components.expand())

    def trigsimp(self):
        if self.rank > 0:
            return Tensor(self.name, self.components.applyfunc(lambda expr: expr.trigsimp()), self.indices)
        else:
            return Tensor(self.name, self.components.trigsimp())

    def subs(self, *args, **kwargs):
        if self.rank > 0:
            return Tensor(self.name, self.components.applyfunc(lambda expr: expr.subs(*args, **kwargs)), self.indices)
        else:
            return Tensor(self.name, self.components.subs(*args, **kwargs))


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

    def covariant_gradient(self):
        spacetime.ensure_christoffel()

        if self.rank == 0:
            return Tensor(f"∇{self.name}",self.partial_gradient())

        if self.rank != 0:
            result = self.partial_gradient()

            for i in range(self.rank):
                new_order = list(range(self.rank + 1))
                new_order.remove(0)
                new_order.insert(i+1, 0)

                if self.indices[i] == 1:
                    result += (spacetime.christoffel * self).contract(1,3+i).reorder_indices(new_order)
                elif self.indices[i] == -1:
                    result -= (spacetime.christoffel * self).contract(0,3+i).reorder_indices(new_order)
                else:
                    raise ValueError(f"Invalid index position {self.indices[i]} encountered at index {i}.")
                result = result.expand()

            return Tensor(f"∇{self.name}", result.trigsimp().simplify(), [-1] + self.indices)
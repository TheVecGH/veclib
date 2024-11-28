import itertools

def permute(x, permute_positions):
	if len(x) < len(permute_positions) or len(set(permute_positions)) != len(permute_positions):
		raise ValueError(f"Invalid positions {permute_positions} to permute list {x}")

	permuted_indices_list = list()

	for perm in itertools.permutations(permute_positions):
	    permuted_indices = list()
	    permuted_index_count = 0
	    for i in x:
	        if i in permute_positions:
	            permuted_indices.append(perm[permuted_index_count])
	            permuted_index_count += 1
	        else:
	            permuted_indices.append(i)
	    permuted_indices_list.append(permuted_indices)	

	return permuted_indices_list

def permutation_sign(perm):
    """
    Calculate the sign of a permutation.
    
    Parameters:
    perm (list): A list representing a permutation of range(len(perm)).
    
    Returns:
    int: +1 for even permutations, -1 for odd permutations.
    """
    inversions = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:  # Count inversions
                inversions += 1
    return 1 if inversions % 2 == 0 else -1
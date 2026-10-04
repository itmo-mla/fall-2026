from .ccv import (
    compact_profile,
    get_2nn_idx_list,
    remove_row_set_compact_profile,
)


def ref_mask(x, y):
    p_old = compact_profile(x, y, 1)

    removed_set = set()

    while x.shape[0] - len(removed_set) > 2:
        nn_idx = get_2nn_idx_list(x, removed_set)

        min_loo = None
        remove_idx = -1
        for i in range(x.shape[0]):
            if i in removed_set:
                continue
            loo = remove_row_set_compact_profile(x, y, i, p_old, removed_set, nn_idx)
            if min_loo is None or loo < min_loo:
                min_loo = loo
                remove_idx = i

        if min_loo is None or min_loo > p_old:
            break

        removed_set |= {remove_idx}
        p_old = min_loo

    return [i for i in range(x.shape[0]) if i not in removed_set]

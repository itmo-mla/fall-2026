from .ccv import compact_profile, remove_row_set_compact_profile


def ref_mask(x, y):
    p_old = compact_profile(x, y, 1)

    removed_set = set()

    while True:
        min_llo = None
        remove_idx = -1
        for i in range(x.shape[0]):
            llo = remove_row_set_compact_profile(x, y, i, p_old, removed_set)
            if min_llo is None:
                min_llo = llo + 1

            if min_llo > llo:
                min_llo = llo
                remove_idx = i
        if min_llo > p_old:
            break
        removed_set = removed_set.union({remove_idx})
    return [i for i in range(x.shape[0]) if i not in removed_set]

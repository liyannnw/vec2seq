#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# from _fast_distance import fast_similitude
import math
import multiprocessing
import sys
import time

###############################################################################

__verbose__ = False
__trace__ = False

# __date__, __version__ = '23/11/2008', '1.0'  # Yves Lepage
# __date__, __version__ = '17/11/2016', '1.1'  # Kaveeta: Python 3 (2.7 compatible)
# __date__, __version__ = '18/11/2016', '1.2'  # Kaveeta: Multi-thread (similarity_multi_thread)
# __date__, __version__ = '20/11/2016', '1.3'  # Kaveeta: Support word unit in Allison and Dix


###############################################################################

def list_of_indices(c, s):
    """
    Returns the list of indices of the occurrences of c in s
    """
    result = []
    i = 0
    while i < len(s):
        if type(s) == list:
            try:
                i = s[i:].index(c) + i + 1
            except ValueError:
                i = 0
        else:
            i = s.find(c, i) + 1

        if 0 != i:
            result.append(i - 1)
        else:
            break
    return result


def look_for_threshold_index(j, threshold, left=None, right=None):
    """
    Look for k such that threshold[k-1] < j <= threshold[k]
    Algorithm: dichotomy search

    >>> look_for_threshold_index(4,[4])
    0
    >>> look_for_threshold_index(4,[0, 1, 2, 3, 4, 5, 6, 7])
    4
    >>> look_for_threshold_index(5,[0, 2, 4, 6, 8, 10, 12, 14])
    3
    """

    if (None, None) == (left, right):
        left, right = 0, len(threshold) - 1

    if left > right:
        raise ValueError('Value in left higher than right')
    elif left + 1 == right or left == right:
        return right
    else:
        mid = int((left + right) / 2)
        if j <= threshold[mid]:
            left, right = left, mid
        else:
            left, right = mid, right
        return look_for_threshold_index(j, threshold, left, right)


###############################################################################

def trace_array(s1, s2, array):
    l1, l2 = len(s1), len(s2)
    s = ''
    s += (' ' * 3)
    s += ''.join((' %s ' % c2) for c2 in s2)
    s += '\n'
    s += '\n'.join((' %s ' % c1) + ''.join(
        ('%3d' % array[i1 + 1][i2 + 1]) for i2 in range(l2)) for i1, c1 in enumerate(s1))
    print(s, file=sys.stderr)


###############################################################################

def similarity_Hunt_and_Szymanski(s1, s2):
    """Return the similarity between two strings,
    i.e., the maximal number of characters in the same order in the two strings
    Algorithm: [Hunt and Szymanski, 1977] in O((|d| + log(r)) x log(min(|s1|,|s2|)))
    where d is the number of different symbols in the longest string
    and r is the number of positions with the same symbol in the two strings (equality points)

    >>> similarity_Hunt_and_Szymanski('','abcd')
    0
    >>> similarity_Hunt_and_Szymanski('abcd','abcd')
    4
    >>> similarity_Hunt_and_Szymanski('abcd','wxyz')
    0
    >>> similarity_Hunt_and_Szymanski('abcd','wxabyd')
    3
    """
    # let s1 be the shortest string
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    equal = {}

    # particular cases
    if '' == s1:
        return 0

    # first preprocessing step: computation of the equality points
    for i in range(0, len(s2)):
        equal[i + 1] = list_of_indices(s2[i], s1)[::-1]

    # second preprocessing step: similarity threshold table
    threshold = [len(s1) + 1 for _ in range(0, len(s2) + 1)]
    threshold[0] = 0
    # processing step: algorithm proper
    for i in range(0, len(s2)):
        for j in equal[i + 1]:
            k = look_for_threshold_index(j, threshold)  # look for k such that threshold[k-1] < j <= threshold[k]:
            if j < threshold[k]:
                threshold[k] = j

    # postprocessing step: looking for the result, i.e., the similarity between the two strings
    # it is the first index in threshold with a value different from len(s1) + 1, starting from the right
    result = 0
    for k in range(len(s2), 0, -1):
        if len(s1) + 1 != threshold[k]:
            result = k
            break
    return result


###############################################################################

def similarity_with_penalty(s1, s2):
    array = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i1 in range(0, len(s1)):
        for i2 in range(0, len(s2)):
            if s1[i1] == s2[i2]:
                if 0 == i1 or 0 == i2 or s1[i1 - 1] == s2[i2 - 1]:
                    r = 2
                    if 0 != i1 and 0 != i2:
                        array[i1 - 1][i2 - 1] += 1
                else:
                    r = 1
            else:
                r = 0
            array[i1 + 1][i2 + 1] = max(array[i1][i2] + r, array[i1 + 1][i2], array[i1][i2 + 1])
    return array[len(s1)][len(s2)]


###############################################################################

def similarity_naive(s1, s2):
    """Return the similarity between two strings,
    i.e., the maximal number of characters in the same order in the two strings
    Standard dynamic programming algorithm in o(|s1| x |s2|)

    >>> similarity_naive('','abcd')
    0
    >>> similarity_naive('abcd','abcd')
    4
    >>> similarity_naive('abcd','abcd') == len('abcd')
    True
    >>> similarity_naive('abcd','wxyz')
    0
    >>> similarity_naive('abcd','wxabyd')
    3
    >>> similarity_naive([''],'this is a second string'.split(' '))
    0
    >>> similarity_naive(''.split(' '),'this is a second string'.split(' '))
    0
    >>> similarity_naive('this is my first string'.split(' '),'this is a second string'.split(' '))
    3
    >>> similarity_naive('','this is a second string')
    0
    """

    array = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i1 in range(0, len(s1)):
        for i2 in range(0, len(s2)):
            if s1[i1] == s2[i2]:
                r = 1
            else:
                r = 0
            array[i1 + 1][i2 + 1] = max(array[i1][i2] + r, array[i1 + 1][i2], array[i1][i2 + 1])
    if __trace__:
        trace_array(s1, s2, array)
    return array[len(s1)][len(s2)]


###############################################################################

def dtw_naive(s1, s2):
    """Return the dynamic time warping between two strings,
    Standard dynamic programming algorithm in o(|s1| x |s2|)

    DTW is NOT verify the separation axiom:
    >>> dtw_naive('a','aa')
    0

    DTW does NOT verity the triangle inequality:
    >>> dtw_naive('abbc','acc')
    2
    >>> dtw_naive('abbc','abc')
    0
    >>> dtw_naive('abc','acc')
    1
    >>> dtw_naive('abbc', 'acc') > dtw_naive('abbc', 'abc') + dtw_naive('abc', 'acc')
    True
    >>> dtw_naive('','abcd')
    4
    >>> dtw_naive('a','aa')
    0
    >>> dtw_naive('abcd','abcd')
    0
    >>> dtw_naive('abcd','wxyz')
    4
    >>> dtw_naive('abcd','wxabyd')
    3
    >>> dtw_naive([''],'this is a second string'.split(' '))
    5
    >>> dtw_naive(''.split(' '),'this is a second string'.split(' '))
    5
    >>> dtw_naive('this is my first string'.split(' '),'this is a second string'.split(' '))
    2
    >>> dtw_naive('','this is a second string')
    23
    """

    l1, l2 = len(s1), len(s2)
    inf = max(l1, l2)
    array = [[0 for _ in range(l2 + 1)] for _ in range(l1 + 1)]
    for i1 in range(l1):
        array[i1 + 1][0] = inf
    for i2 in range(l2):
        array[0][i2 + 1] = inf
    for i1 in range(l1):
        for i2 in range(l2):
            if s1[i1] == s2[i2]:
                r = 0
            else:
                r = 1
            array[i1 + 1][i2 + 1] = r + min(array[i1][i2], array[i1 + 1][i2], array[i1][i2 + 1])
    if __trace__:
        trace_array(s1, s2, array)
    return array[len(s1)][len(s2)]


###############################################################################

def similarity(s1, s2, method='Hunt and Szymanski', word_unit=False):
    assert type(s1) == type(s2), 'arguments of distance should be of same type'

    # For word unit, split string to list of words
    if word_unit and type(s1) != list and type(s2) != list:
        s1, s2 = s1.split(' '), s2.split(' ')

    # choose the computation algorithm
    if method == 'naive':
        return similarity_naive(s1, s2)
    elif method == 'Hunt and Szymanski':
        return similarity_Hunt_and_Szymanski(s1, s2)
    elif method == 'dtw':
        return dtw_naive(s1, s2)
    elif method == 'Dix and Allison in C':
        # TODO: directly implement hashing in fast_distance.c directly instead?
        # In word unit case, encode list to char first
        if type(s1) and type(s2) == list:
            # Character start at 1, hence +1 for id
            word_hash = {word: word_id+1 for word_id, word in enumerate(set(s1))}
            s1 = ''.join([chr(word_hash[word]) for word in s1])
            s2 = ''.join([chr(word_hash[word]) for word in s2 if word in word_hash])
        return fast_similitude(s1, s2)


def similarity_worker(line, method='Hunt and Szymanski', word_unit=False):
    """
    Calculate similarity between pair of string
    """
    result = []
    for s1, s2 in line:
        assert type(s1) == type(s2), 'arguments of distance should be of same type'

        # make similarity available for any list of objects
        if word_unit:
            s1, s2 = s1.split(' '), s2.split(' ')

        # choose the computation algorithm
        if method == 'naive':
            result.append(similarity_naive(s1, s2))
        elif method == 'Hunt and Szymanski':
            result.append(similarity_Hunt_and_Szymanski(s1, s2))
        elif method == 'dtw':
            result.append(dtw_naive(s1, s2))
        elif method == 'Dix and Allison in C':
            result.append(fast_similitude(s1, s2))

    return result


def similarity_multi_thread(pairs, method='Hunt and Szymanski', word_unit=False):
    """
    Perform similarity in multi-thread
    :param pairs: list of pair of sentences
    :param method: similarity algorithms
    :param word_unit: using character-level or word-level
    :return: list of similarity values
    """
    # Split jobs
    split_size = int(math.ceil(len(pairs) / options.thread))
    pair_chunks = [pairs[i:min(len(pairs), i + split_size)] for i in range(0, len(pairs), split_size)]

    # Start calculation threads
    pool = multiprocessing.Pool(processes=options.thread)
    result = []
    for chunk in pair_chunks:
        result.append(pool.apply_async(similarity_worker, (chunk, method, word_unit)))
    pool.close()
    pool.join()

    # Merge result
    results = []
    [results.extend(res.get()) for res in result]

    return results


###############################################################################

def distance(s1, s2, method='Hunt and Szymanski', word_unit=False):
    """
    Return the canonical distance between two strings,
    i.e., the minimal number of insertions and deletions necessary to transform one string into the other one
    This is different form the Levenshtein distance, where substitutions count.
    For the canonical distance, a substitution can be said to count for 2 operations:
    a deletion followed by an insertion.

    >>> distance('abcd','abcd')
    0
    >>> distance('abcd','wxyz')
    8
    >>> distance('abcd','wxabyd')
    4
    >>> distance('','this is a second string')
    23
    >>> distance('this is my first string','this is a second string')
    12

    This is the same as 'word_unit=True'
    >>> distance('this is my first string'.split(' '),'this is a second string'.split(' '))
    4
    >>> distance('this is my first string','this is a second string', word_unit=True)
    4
    >>> distance('this is my first string','this is a second string', method='naive', word_unit=True)
    4
    >>> distance('this is my first string','this is a second string', method='dtw', word_unit=True)
    6
    >>> distance('this is my first string','this is a second string', method='Dix and Allison in C', word_unit=True)
    4
    """

    assert type(s1) == type(s2), 'arguments of distance should be of same type'
    if word_unit:
        s1, s2 = s1.split(' '), s2.split(' ')
    return len(s1) + len(s2) - 2 * similarity(s1, s2, method)


###############################################################################

def read_argv():
    from optparse import OptionParser
    this_version = '(c) 23/11/2008 Yves Lepage'
    this_description = 'Distance and similarity computation'
    this_usage = 'usage: %prog [options]\n'

    parser = OptionParser(version=this_version, description=this_description, usage=this_usage)
    parser.add_option('-t', '--trace',
                      action='store_true', dest='trace', default=False,
                      help='trace the program')
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose', default=False,
                      help='give some information')
    parser.add_option('-j', '--thread',
                      action='store', type=int, dest='thread', default=multiprocessing.cpu_count(),
                      help='number of thread to run')

    return parser.parse_args()


###############################################################################

def main():
    n = 10000
    strings = [['toto', 'tata'],
               ['louis-philippe', 'poire'],
               ['ceci est un exemple plus parlant', 'voilà un autre exemple'],
               ['here is another possibly longer example', 'and yet another example that is long too']]

    # 1.General test
    print('1.General test')
    for i in range(len(strings)):
        for method in ('naive', 'dtw', 'Hunt and Szymanski', 'Dix and Allison in C'):
            t1 = time.time()
            for _ in range(int(n)):
                result = similarity(strings[i][0], strings[i][1], method)
            t2 = time.time()
            print('%0.3f ms for (%s, %s) = %d computed %d times with %s' %
                  ((t2 - t1) * 1000, strings[i][0], strings[i][1], result, n, method))
        print()

    # 2.Single thread test (Baseline)
    n = 1000000
    method = 'Dix and Allison in C'
    print('2.Run single thread %d times with %s' % (n, method))
    t1 = time.time()
    for i in range(len(strings)):
        for _ in range(int(n)):
            result = similarity(strings[i][0], strings[i][1], method)
        print('(%s, %s) = %d computed %d times with %s' %
              (strings[i][0], strings[i][1], result, n, method))
    t2 = time.time()
    print('Single thread take %0.3f s' % (t2 - t1))
    print()

    # 3.Multi-thread test
    print('3.Run %d threads %d times with %s' % (options.thread, n, method))
    t1 = time.time()

    # Generate list of all question pair
    pairs = []
    [pairs.extend([pair for pair in strings]) for _ in range(n)]
    results = similarity_multi_thread(pairs, method)

    # Print out result
    for i in range(len(strings)):
        print('(%s, %s) = %d computed %d times with %s' %
              (strings[i][0], strings[i][1], results[i], int(n), method))

    t2 = time.time()
    print('Multi thread take %0.3f s' % (t2 - t1))
    print()

    # 4.Word unit test
    n = 10000
    strings = [['ceci est un exemple plus parlant', 'voilà un autre exemple'],
               ['here is another possibly longer example', 'and yet another example that is long too'],
               ['here is example sentence which are quite long and possibly slow some algorithms down',
                'and yet another example that is quite long too might be a bit slow for some algorithms']]
    print('4.Word unit test')
    for i in range(len(strings)):
        for method in ('naive', 'dtw', 'Hunt and Szymanski', 'Dix and Allison in C'):
            t1 = time.time()
            for _ in range(int(n)):
                result = similarity(strings[i][0], strings[i][1], method, word_unit=True)
            t2 = time.time()
            print('%0.3f ms for (%s, %s) = %d computed %d times with %s' %
                  ((t2 - t1) * 1000, strings[i][0], strings[i][1], result, n, method))
        print()

if __name__ == '__main__':
    options, args = read_argv()
    __trace__ = options.trace
    __verbose__ = options.verbose
    main()

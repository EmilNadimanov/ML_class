#!/usr/bin/env python3
class Tree:
    '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.

  Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
  '''

    def leaf(data):
        '''Create a leaf tree
    '''
        return Tree(data=data)

    # pretty-print trees
    def __repr__(self):
        if self.is_leaf():
            return "Leaf(%r)" % self.data
        else:
            return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right)

    # all arguments after `*` are *keyword-only*!
    def __init__(self, *, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        '''Check if this tree is a leaf tree
    '''
        return self.left == None and self.right == None

    def children(self):
        '''List of child subtrees
    '''
        return [x for x in [self.left, self.right] if x]

    def depth(self):
        '''Compute the depth of a tree
    A leaf is depth-1, and a child is one deeper than the parent.
    '''
        return max([x.depth() for x in self.children()], default=0) + 1

if __name__ == "__main__":
    root = Tree(data="isSystems?")
    takenOtherSys = Tree(data="takenOtherSys")
    morning = Tree(data="morning?")
    likedOtherSys = Tree(data="likedOtherSys?")
    l1 = Tree.leaf(data="like")
    l2 = Tree.leaf(data="like")
    l3 = Tree.leaf(data="nah")
    l4 = Tree.leaf(data="nah")
    l5 = Tree.leaf(data="like")
    root.left = l1
    root.right = takenOtherSys
    takenOtherSys.left = morning
    takenOtherSys.right = likedOtherSys
    morning.left = l2
    morning.right = l3
    likedOtherSys.left = l4
    likedOtherSys.right = l5
    print(root)

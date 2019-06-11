package Code;

import Data.TreeNode;

import java.util.Stack;

public class BSTIterator {

    private Stack<TreeNode> stack;

    public BSTIterator(TreeNode root) {
        stack = new Stack<>();
        TreeNode p = root;
        while (p != null) {
            stack.push(p);
            p = p.left;
        }
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        if (hasNext()) {
            TreeNode pop = stack.pop();
            TreeNode p = pop.right;
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            return pop.val;
        }
        return -1;
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return !stack.empty();
    }
}
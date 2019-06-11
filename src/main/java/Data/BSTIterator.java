package Data;

import java.util.Stack;

public class BSTIterator {
    private Stack<TreeNode> stack = new Stack<>();

    public BSTIterator(TreeNode root) {
        if (root != null) {
            stack.push(root);
            TreeNode node = root;
            while (node.left != null) {
                stack.push(node.left);
                node = node.left;
            }
        }
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return !stack.isEmpty();
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        if (hasNext()) {
            TreeNode node = stack.pop();
            int t = node.val;
            node = node.right;
            if (node != null) {
                stack.push(node);
                while (node.left != null) {
                    stack.push(node.left);
                    node = node.left;
                }
            }
            return t;
        } else {
            return -1;
        }
    }
}

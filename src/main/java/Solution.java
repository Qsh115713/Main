import Data.TreeNode;

import java.util.*;

public class Solution {

    public void recoverTree(TreeNode root) {
        Stack<TreeNode> s = new Stack<>();
        TreeNode p = root, prev = null, tmp = null;
        while (p != null || !s.empty()) {
            while (p != null) {
                s.push(p);
                p = p.left;
            }
            p = s.pop();
            if (prev != null && prev.val >= p.val) {
                if (tmp != null) {
                    int t = tmp.val;
                    tmp.val = p.val;
                    p.val = t;
                    break;
                } else {
                    tmp = prev;
                }
            }
            prev = p;
            p = p.right;
        }
    }

    public boolean isValidBST(TreeNode root) {
        Stack<TreeNode> s = new Stack<>();
        TreeNode p = root;
        boolean tag = true;
        int prev = 0;
        while (p != null || !s.empty()) {
            while (p != null) {
                s.push(p);
                p = p.left;
            }
            p = s.pop();
            if (tag || prev < p.val) {
                prev = p.val;
                tag = false;
            } else {
                return false;
            }
            p = p.right;
        }
        return true;
    }

    public int numTrees(int n) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        return numTreesLoop(map, 1, n);
    }

    private int numTreesLoop(Map<Integer, Integer> map, int lo, int hi) {
        if (lo >= hi) return 1;
        int k = hi - lo + 1;
        if (map.containsKey(k)) return map.get(k);
        int left, right, res = 0;
        for (int i = lo; i <= hi; i++) {
            left = numTreesLoop(map, lo, i - 1);
            right = numTreesLoop(map, i + 1, hi);
            res += left * right;
            map.put(i - lo, left);
            map.put(hi - i, right);
        }
        return res;
    }

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return new ArrayList<>();
        return generateTreesLoop(1, n);
    }

    private List<TreeNode> generateTreesLoop(int lo, int hi) {
        List<TreeNode> res = new ArrayList<>();
        if (lo > hi) {
            res.add(null);
            return res;
        }
        List<TreeNode> left, right;
        for (int i = lo; i <= hi; i++) {
            left = generateTreesLoop(lo, i - 1);
            right = generateTreesLoop(i + 1, hi);
            for (TreeNode lNode : left) {
                for (TreeNode rNode : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = lNode;
                    root.right = rNode;
                    res.add(root);
                }
            }
        }
        return res;
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> s = new Stack<>();
        s.push(root);
        s.push(root);
        while (!s.empty()) {
            TreeNode p = s.pop();
            if (s.empty() || p.val != s.peek().val) {
                res.add(p.val);
                continue;
            }
            if (p.right != null) {
                s.push(p.right);
                s.push(p.right);
            }
            if (p.left != null) {
                s.push(p.left);
                s.push(p.left);
            }
        }
        return res;
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> s = new Stack<>();
        TreeNode p = root;
        while (p != null || !s.empty()) {
            while (p != null) {
                res.add(p.val);
                s.push(p);
                p = p.left;
            }
            p = s.pop();
            p = p.right;
        }
        return res;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> s = new Stack<>();
        TreeNode p = root;
        while (p != null || !s.empty()) {
            while (p != null) {
                s.push(p);
                p = p.left;
            }
            p = s.pop();
            res.add(p.val);
            p = p.right;
        }
        return res;
    }
}

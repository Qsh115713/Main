import Data.TreeNode;

import java.util.*;

public class Solution {

    public int maxPathSum(TreeNode root) {
        if (root == null) return 0;
        maxPathSumLoop(root);
        return maxVal;
    }

    private int maxVal = Integer.MIN_VALUE;

    private int maxPathSumLoop(TreeNode root) {
        if (root == null) return 0;
        int left = Math.max(maxPathSumLoop(root.left), 0);
        int right = Math.max(maxPathSumLoop(root.right), 0);
        maxVal = Math.max(maxVal, root.val + left + right);
        return root.val + Math.max(left, right);
    }

    public void flatten(TreeNode root) {
        TreeNode cur = root;
        while (cur != null) {
            if (cur.left != null) {
                TreeNode pre = cur.left;
                while (pre.right != null) {
                    pre = pre.right;
                }
                pre.right = cur.right;
                cur.right = cur.left;
                cur.left = null;
            }
            cur = cur.right;
        }
    }

    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        return Math.min(left == 0 ? Integer.MAX_VALUE : left, right == 0 ? Integer.MAX_VALUE : right) + 1;
    }

    public boolean isBalanced(TreeNode root) {
        return getHeight(root) != -1;
    }

    private int getHeight(TreeNode root) {
        if (root == null) return 0;
        int leftHeight = getHeight(root.left);
        if (leftHeight == -1) return -1;
        int rightHeight = getHeight(root.right);
        if (rightHeight == -1) return -1;
        if (Math.abs(leftHeight - rightHeight) > 1) return -1;
        return Math.max(leftHeight, rightHeight) + 1;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTLoop(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTLoop(int[] nums, int lo, int hi) {
        if (lo > hi) return null;
        int mid = lo + (hi - lo) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBSTLoop(nums, lo, mid - 1);
        root.right = sortedArrayToBSTLoop(nums, mid + 1, hi);
        return root;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;
    }

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

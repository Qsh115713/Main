package Code;

import Data.*;

import java.util.*;

public class LeetCode {



    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        if (s == null || s.equals("") || s.length() < p.length()) {
            return res;
        }
        Map<Character, Integer> map = getDefaultMap(p);
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (!map.containsKey(ch)) {
                map = getDefaultMap(p);
                continue;
            }
            int count = map.get(ch) - 1;
            if (count == 0) {
                map.remove(ch);
            } else {
                map.put(ch, count);
            }
            //TODO
        }
        return res;
    }

    private Map<Character, Integer> getDefaultMap(String p) {
        Map<Character, Integer> map = new HashMap<>();
        for (char ch : p.toCharArray()) {
            map.put(ch, map.getOrDefault(ch, 0) + 1);
        }
        return map;
    }

    public int pathSum(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        return pathSumHelper(root, sum, 0) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }

    private int pathSumHelper(TreeNode root, int sum, int cur) {
        if (root == null) {
            return 0;
        }
        cur += root.val;
        int res = cur == sum ? 1 : 0;
        res += pathSumHelper(root.left, sum, cur);
        res += pathSumHelper(root.right, sum, cur);
        return res;
    }

    public int multiBag(int[] w, int[] v, int[] c, int m) {
        int n = 0;
        for (int i = 1; i < c.length; i++) {
            n += c[i];
        }
        int[] w1 = new int[n + 1];
        int[] v1 = new int[n + 1];
        int k = 1;
        for (int i = 1; i < w.length; i++) {
            while (c[i]-- > 0) {
                w1[k] = w[i];
                v1[k] = v[i];
                k++;
            }
        }
        return zeroOneBag(w1, v1, m);
    }

    public int completeBag(int[] w, int[] v, int m) {
        int[] dp = new int[m + 1];
        for (int i = 1; i < w.length; i++) {
            for (int j = w[i]; j <= m; j++) {
                dp[j] = Math.max(dp[j], dp[j - w[i]] + v[i]);
            }
        }
        return dp[m];
    }

    public int zeroOneBag(int[] w, int[] v, int m) {
        int[] dp = new int[m + 1];
        for (int i = 1; i < w.length; i++) {
            for (int j = m; j >= w[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - w[i]] + v[i]);
            }
        }
        return dp[m];
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        LinkedList<Integer>[] neibour = new LinkedList[numCourses];
        Queue<Integer> queue = new ArrayDeque<>();
        int[] indegree = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            neibour[i] = new LinkedList<>();
        }
        for (int[] item : prerequisites) {
            neibour[item[1]].add(item[0]);
            indegree[item[0]]++;
        }
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) queue.add(i);
        }
        int index = 0;
        int[] res = new int[numCourses];
        while (!queue.isEmpty()) {
            int p = queue.poll();
            res[index++] = p;
            for (int item : neibour[p]) {
                if (--indegree[item] == 0) queue.add(item);
            }
        }
        return index == numCourses ? res : new int[0];
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] in = new int[numCourses];
        for (int[] item : prerequisites) {
            in[item[1]]++;
        }
        Queue<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) {
            if (in[i] == 0) q.add(i);
        }
        int res = 0;
        while (!q.isEmpty()) {
            int p = q.poll();
            res++;
            for (int[] item : prerequisites) {
                if (item[0] == p) {
                    in[item[1]]--;
                    if (in[item[1]] == 0) q.add(item[1]);
                }
            }
        }
        return res == numCourses;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> s = new Stack<>();
        Stack<TreeNode> t = new Stack<>();
        s.push(root);
        int level = 0;
        while (!s.empty()) {
            res.add(new ArrayList<>());
            while (!s.empty()) {
                TreeNode node = s.pop();
                if (level % 2 == 0) {
                    if (node.left != null) t.push(node.left);
                    if (node.right != null) t.push(node.right);
                }
                if (level % 2 == 1) {
                    if (node.right != null) t.push(node.right);
                    if (node.left != null) t.push(node.left);
                }
                res.get(res.size() - 1).add(node.val);
            }
            while (!t.empty()) {
                s.push(t.pop());
            }
            level++;
        }
        return res;
    }

    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        levelOrderHelper(res, root, 0);
        return res;
    }

    private void levelOrderHelper(List<List<Integer>> res, TreeNode root, int depth) {
        if (res.size() <= depth) res.add(new ArrayList<>());
        if (root.left != null) levelOrderHelper(res, root.left, depth + 1);
        if (root.right != null) levelOrderHelper(res, root.right, depth + 1);
        res.get(depth).add(root.val);
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> s = new Stack<>();
        Stack<TreeNode> t = new Stack<>();
        s.push(root);
        while (!s.empty()) {
            res.add(new ArrayList<>());
            while (!s.empty()) {
                TreeNode node = s.pop();
                if (node.left != null) t.push(node.left);
                if (node.right != null) t.push(node.right);
                res.get(res.size() - 1).add(node.val);
            }
            while (!t.empty()) {
                s.push(t.pop());
            }
        }
        return res;
    }

    public boolean isSubtree(TreeNode s, TreeNode t) {
        boolean res = false;
        if (s != null && t != null) {
            if (s.val == t.val) res = isSubtreeHelper(s, t);
            res |= isSubtree(s.left, t);
            res |= isSubtree(s.right, t);
        }
        return res;
    }

    private boolean isSubtreeHelper(TreeNode s, TreeNode t) {
        if (t == null) return s == null;
        if (s == null || s.val != t.val) return false;
        return isSubtreeHelper(s.left, t.left) && isSubtreeHelper(s.right, t.right);
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode root = new ListNode(0), p = root, q, r = null, t;
        root.next = head;
        q = p.next;
        int i = 1;
        while (q != null) {
            if (i < m) {
                p = p.next;
                q = q.next;
            } else if (i <= n) {
                if (i == m) {
                    p.next = null;
                    r = q;
                }
                t = q;
                q = q.next;
                t.next = p.next;
                p.next = t;
            } else {
                if (r != null) r.next = q;
                break;
            }
            i++;
        }
        return root.next;
    }

    public ListNode reverseList(ListNode head) {
        ListNode root = new ListNode(0), p = head, q;
        while (p != null) {
            q = p;
            p = p.next;
            q.next = root.next;
            root.next = q;
        }
        return root.next;
    }
}

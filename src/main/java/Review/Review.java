package Review;

import Data.*;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Review {

    public int findDuplicate(int[] nums) {
        if (nums.length <= 1) return 0;
        int slow = nums[0];
        int fast = nums[nums[0]];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        fast = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    public int longestConsecutive(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int left, right, res = 0;
        for (int num : nums) {
            if (map.containsKey(num)) continue;
            left = map.getOrDefault(num - 1, 0);
            right = map.getOrDefault(num + 1, 0);
            res = Math.max(res, left + right + 1);
            map.put(num, left + right + 1);
            map.put(num - left, left + right + 1);
            map.put(num + right, left + right + 1);
        }
        return res;
    }

    public Node connect(Node root) {
        Node head = new Node(), p = head, t = root;
        while (t != null) {
            if (t.left != null) {
                p.next = t.left;
                p = p.next;
            }
            if (t.right != null) {
                p.next = t.right;
                p = p.next;
            }
            t = t.next;
            if (t == null) {
                t = head.next;
                head.next = null;
                p = head;
            }
        }
        return root;
    }

    public TreeNode sortedListToBST(ListNode head) {
        int len = getListLength(head);
        this.head = head;
        return sortedListToBSTHelper(0, len - 1);
    }

    private ListNode head;

    private TreeNode sortedListToBSTHelper(int lo, int hi) {
        if (lo > hi) return null;
        int mid = lo + (hi - lo) / 2;
        TreeNode left = sortedListToBSTHelper(lo, mid - 1);
        TreeNode root = new TreeNode(head.val);
        root.left = left;
        head = head.next;
        root.right = sortedListToBSTHelper(mid + 1, hi);
        return root;
    }

    private int getListLength(ListNode head) {
        ListNode p = head;
        int len = 0;
        while (p != null) {
            ++len;
            p = p.next;
        }
        return len;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTHelper(int[] nums, int lo, int hi) {
        if (lo > hi) return null;
        int mid = lo + (hi - lo) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBSTHelper(nums, lo, mid - 1);
        root.right = sortedArrayToBSTHelper(nums, mid + 1, hi);
        return root;
    }

    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> s = new Stack<>();
        Stack<TreeNode> t = new Stack<>();
        s.push(root);
        while (!s.empty()) {
            res.add(new ArrayList<>());
            while (!s.empty()) {
                t.push(s.pop());
            }
            while (!t.empty()) {
                TreeNode temp = t.pop();
                if (temp.left != null) s.push(temp.left);
                if (temp.right != null) s.push(temp.right);
                res.get(res.size() - 1).add(temp.val);
            }
        }
        return res;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
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

    public boolean isSymmetric(TreeNode root) {
        return root == null || isSymmetricHelper(root.left, root.right);
    }

    private boolean isSymmetricHelper(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        return left.val == right.val && isSymmetricHelper(left.left, right.right) && isSymmetricHelper(left.right, right.left);
    }

    public int minDiffInBST(TreeNode root) {
        Stack<TreeNode> s = new Stack<>();
        TreeNode pre = null;
        int res = Integer.MAX_VALUE;
        while (root != null || !s.empty()) {
            while (root != null) {
                s.push(root);
                root = root.left;
            }
            root = s.pop();
            if (pre != null && root.val - pre.val < res) {
                res = root.val - pre.val;
            }
            if (res == 1) return res;
            pre = root;
            root = root.right;
        }
        return res;
    }

    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> s = new Stack<>();
        while (root != null || !s.empty()) {
            while (root != null) {
                s.push(root);
                root = root.left;
            }
            root = s.pop();
            if (--k == 0) return root.val;
            root = root.right;
        }
        return 0;
    }

    public boolean isValidBST(TreeNode root) {
        if (root == null) return true;
        Stack<TreeNode> s = new Stack<>();
        TreeNode pre = null, p = root;
        while (p != null || !s.empty()) {
            while (p != null) {
                s.push(p);
                p = p.left;
            }
            p = s.pop();
            if (pre != null && pre.val >= p.val) return false;
            pre = p;
            p = p.right;
        }
        return true;
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> s = new Stack<>();
        if (root == null) return res;
        s.push(root);
        s.push(root);
        while (!s.empty()) {
            TreeNode p = s.pop();
            if (!s.empty() && p.val == s.peek().val) {
                if (p.right != null) {
                    s.push(p.right);
                    s.push(p.right);
                }
                if (p.left != null) {
                    s.push(p.left);
                    s.push(p.left);
                }
            } else {
                res.add(p.val);
            }
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
            if (!s.empty()) {
                p = s.pop();
                res.add(p.val);
                p = p.right;
            }
        }
        return res;
    }

    public int numDecodings(String s) {
        if (s == null || s.length() == 0) return 0;
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) != '0' ? 1 : 0;
        for (int i = 2; i <= n; i++) {
            int one = Integer.parseInt(s.substring(i - 1, i));
            int two = Integer.parseInt(s.substring(i - 2, i));
            if (one >= 1 && one <= 9) dp[i] += dp[i - 1];
            if (two >= 10 && two <= 26) dp[i] += dp[i - 2];
        }
        return dp[n];
    }

    public void sortColors(int[] nums) {
        int i = 0, j = nums.length - 1, k = 0;
        while (k <= j) {
            if (nums[k] == 0) {
                swap(nums, i++, k++);
            } else if (nums[k] == 1) {
                k++;
            } else {
                swap(nums, k, j--);
            }
        }
    }

    public void setZeroes(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) return;
        int m = matrix.length, n = matrix[0].length;
        boolean tag = false;
        for (int i = 0; i < m; i++) {
            if (!tag && matrix[i][0] == 0) tag = true;
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (matrix[0][0] == 0) {
            for (int j = 1; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
        if (tag) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    public int fib(int N) {
        if (N == 0) return 0;
        int[] dp = new int[N + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= N; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[N];
    }

    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        if (n == 0) return 0;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = cost[0];
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1], dp[i - 2]) + cost[i - 1];
        }
        return Math.min(dp[n - 1], dp[n]);
    }

    public int climbStairs(int n) {
        if (n < 1) return 0;
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int minPathSum(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) return 0;
        int m = grid.length, n = grid[0].length;
        int[] dp = new int[m];
        for (int i = 0; i < m; i++) {
            dp[i] = (i == 0 ? 0 : dp[i - 1]) + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            dp[0] += grid[0][j];
            for (int i = 1; i < m; i++) {
                dp[i] = Math.min(dp[i - 1], dp[i]) + grid[i][j];
            }
        }
        return dp[m - 1];
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid.length == 0 || obstacleGrid[0].length == 0) return 0;
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[] dp = new int[m];
        for (int i = 0; i < n; i++) {
            if (i != 0 && dp[i - 1] == 0) break;
            if (obstacleGrid[i][0] != 1) dp[i] = 1;
        }
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[0][j] == 1) dp[0] = 0;
            for (int i = 1; i < m; i++) {
                dp[i] = obstacleGrid[i][j] == 1 ? 0 : dp[i] + dp[i - 1];
            }
        }
        return dp[m - 1];
    }

    public int uniquePaths(int m, int n) {
        int[] dp = new int[m];
        dp[0] = 1;
        for (int j = 0; j < n; j++) {
            for (int i = 1; i < m; i++) {
                dp[i] += dp[i - 1];
            }
        }
        return dp[m - 1];
    }

    public List<Interval> merge(List<Interval> intervals) {
        if (intervals == null || intervals.size() <= 1) return intervals;
        intervals.sort((Interval o1, Interval o2) -> Integer.compare(o1.start, o2.start));
        List<Interval> res = new ArrayList<>();
        Interval temp = intervals.get(0);
        for (int i = 1; i < intervals.size(); i++) {
            if (intervals.get(i).start <= temp.end) {
                temp.end = Math.max(temp.end, intervals.get(i).end);
            } else {
                res.add(temp);
                temp = intervals.get(i);
            }
        }
        res.add(temp);
        return res;
    }

    public boolean canJump(int[] nums) {
        int pre = 0, cur = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > pre) {
                pre = cur;
                if (pre < i) return false;
            }
            cur = Math.max(cur, i + nums[i]);
        }
        return true;
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        permuteHelper(res, nums, 0);
        return res;
    }

    private void permuteHelper(List<List<Integer>> res, int[] nums, int index) {
        if (index == nums.length) {
            List<Integer> list = new ArrayList<>();
            for (Integer item : nums) list.add(item);
            res.add(list);
            return;
        }
        for (int i = index; i < nums.length; i++) {
            swap(nums, i, index);
            permuteHelper(res, nums, index + 1);
            swap(nums, i, index);
        }
    }

    public int firstMissingPositive(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            if (nums[i] == i + 1 || nums[i] > nums.length || nums[i] <= 0) i++;
            else if (nums[nums[i] - 1] != nums[i]) swap(nums, i, nums[i] - 1);
            else i++;
        }
        i = 0;
        while (i < nums.length && nums[i] == i + 1) i++;
        return i + 1;
    }

    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    public boolean isValidSudoku(char[][] board) {
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                char number = board[i][j];
                if (number != '.')
                    if (!seen.add(number + " in row " + i) ||
                            !seen.add(number + " in column " + j) ||
                            !seen.add(number + " in block " + i / 3 + "-" + j / 3))
                        return false;
            }
        }
        return true;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.length, Comparator.comparing(ListNode::getVal));
        //PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.length, (ListNode o1, ListNode o2) -> Integer.compare(o1.val, o2.val));
        ListNode head = new ListNode(0), tail = head;
        for (ListNode node : lists) {
            if (node != null) queue.add(node);
        }
        while (queue.size() != 0) {
            tail.next = queue.poll();
            tail = tail.next;
            if (tail.next != null) queue.add(tail.next);
        }
        return head.next;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode l = new ListNode(0), res = l;
        while (l1 != null || l2 != null) {
            if (l2 == null || (l1 != null && l1.val < l2.val)) {
                l.next = l1;
                l1 = l1.next;
            } else {
                l.next = l2;
                l2 = l2.next;
            }
            l = l.next;
        }
        return res.next;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode res = new ListNode(0);
        res.next = head;
        ListNode pre = res, p = res;
        while (p.next != null) {
            if (n <= 0) pre = pre.next;
            p = p.next;
            --n;
        }
        pre.next = pre.next.next;
        return res.next;
    }

    public boolean isMatch2(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = i > 0 && dp[i - 1][j] || dp[i][j - 1];
                } else {
                    dp[i][j] = i > 0 && dp[i - 1][j - 1] &&
                            (p.charAt(j - 1) == '?' || p.charAt(j - 1) == s.charAt(i - 1));
                }
            }
        }
        return dp[m][n];
    }

    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2] || (i > 0 && dp[i - 1][j] && (p.charAt(j - 2) == '.' || p.charAt(j - 2) == s.charAt(i - 1)));
                } else {
                    dp[i][j] = i > 0 && dp[i - 1][j - 1] && (p.charAt(j - 1) == '.' || p.charAt(j - 1) == s.charAt(i - 1));
                }
            }
        }
        return dp[m][n];
    }

    public boolean isPalindrome(ListNode head) {
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        if (fast != null) {
            slow = slow.next;
        }
        slow = reverse(slow);
        fast = head;
        while (slow != null) {
            if (fast.val != slow.val) return false;
            fast = fast.next;
            slow = slow.next;
        }
        return true;
    }

    private ListNode reverse(ListNode head) {
        ListNode prev = null, next;
        while (head != null) {
            next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }

    public static boolean isPalindrome(int x) {
        String str = Integer.toString(x);
        String reverse = new StringBuffer(str).reverse().toString();
        return str.equalsIgnoreCase(reverse);
    }

    public int countSubstrings(String s) {
        int n = s.length(), res = n;
        boolean[][] dp = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = true;
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = i > j - 2 || dp[i + 1][j - 1];
                    if (dp[i][j]) res++;
                }
            }
        }
        return res;
    }

    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    public String shortestPalindrome(String s) {
        int j = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) == s.charAt(j)) ++j;
        }
        if (j == s.length()) return s;
        String suffix = s.substring(j);
        return new StringBuffer(suffix).reverse().toString() + shortestPalindrome(s.substring(0, j)) + suffix;
    }

    public String longestPalindrome(String s) {
        if (s.length() == 0) return "";
        int lo = 0, hi = 0;
        for (int i = 0; i < s.length(); i++) {
            int inc = getLength(s, i, i);
            int exc = getLength(s, i, i + 1);
            int len = Math.max(inc, exc);
            if (len > hi - lo) {
                lo = i - (len - 1) / 2;
                hi = i + len / 2;
            }
        }
        return s.substring(lo, hi + 1);
    }

    private int getLength(String s, int lo, int hi) {
        while (lo >= 0 && hi < s.length() && s.charAt(lo) == s.charAt(hi)) {
            --lo;
            ++hi;
        }
        return hi - lo - 1;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n1 = nums1.length, n2 = nums2.length, n = (n1 + n2) / 2;
        int i = 0, j = 0, k = 0, a = 0, b = 0;
        while (i < n1 || j < n2) {
            a = b;
            if (j >= n2 || (i < n1 && nums1[i] <= nums2[j])) {
                b = nums1[i++];
            } else {
                b = nums2[j++];
            }
            if (k++ == n) break;
        }
        if ((n1 + n2) % 2 == 0) return (a + b) / 2.0;
        return b;
    }

    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int res = 0;
        for (int i = 0, j = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            if (map.containsKey(c)) {
                j = Math.max(j, map.get(c) + 1);
            }
            map.put(c, i);
            res = Math.max(res, i - j + 1);
        }
        return res;
    }

    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor2(root.left, p, q);
        TreeNode right = lowestCommonAncestor2(root.right, p, q);
        return left == null ? right : right == null ? left : root;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return root;
        if (p.val > q.val) {
            TreeNode t = p;
            p = q;
            q = t;
        }
        if (p.val > root.val) return lowestCommonAncestor(root.right, p, q);
        if (q.val < root.val) return lowestCommonAncestor(root.left, p, q);
        return root;
    }

    public boolean findTarget2(TreeNode root, int k) {
        List<Integer> nums = new ArrayList<>();
        inorder(root, nums);
        for (int i = 0, j = nums.size() - 1; i < j; ) {
            if (nums.get(i) + nums.get(j) == k) return true;
            if (nums.get(i) + nums.get(j) < k) i++;
            else j--;
        }
        return false;
    }

    private void inorder(TreeNode root, List<Integer> nums) {
        if (root == null) return;
        inorder(root.left, nums);
        nums.add(root.val);
        inorder(root.right, nums);
    }

    public boolean findTarget(TreeNode root, int k) {
        Set<Integer> set = new HashSet<>();
        return findTargetHelper(root, set, k);
    }

    private boolean findTargetHelper(TreeNode root, Set<Integer> set, int k) {
        if (root == null) return false;
        if (set.contains(k - root.val)) return true;
        set.add(root.val);
        return findTargetHelper(root.left, set, k) || findTargetHelper(root.right, set, k);
    }

    public int subarraySum2(int[] nums, int k) {
        int sum = 0, res = 0;
        Map<Integer, Integer> preSum = new HashMap<>();
        preSum.put(0, 1);
        for (int num : nums) {
            sum += num;
            if (preSum.containsKey(sum - k)) res += preSum.get(sum - k);
            preSum.put(sum, preSum.getOrDefault(sum, 0) + 1);
        }
        return res;
    }

    public int subarraySum(int[] nums, int k) {
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            res += subarraySumHelper(nums, k, i);
        }
        return res;
    }

    private int subarraySumHelper(int[] nums, int r, int i) {
        if (i >= nums.length) return 0;
        int res = 0;
        if (r - nums[i] == 0) ++res;
        return res + subarraySumHelper(nums, r - nums[i], i + 1);
    }
}

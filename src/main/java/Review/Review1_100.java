package Review;

import Data.*;
import javafx.util.Pair;

import java.util.*;

public class Review1_100 {



    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    //***
    public void preorderMorrisTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        TreeNode cur = root, prev;
        while (cur != null) {
            if (cur.left == null) {
                res.add(cur.val);
                cur = cur.right;
            } else {
                // find inorder prev
                prev = cur.left;
                while (prev.right != null && prev.right != cur) {
                    prev = prev.right;
                }
                if (prev.right == null) {
                    res.add(cur.val);
                    prev.right = cur;
                    cur = cur.left;
                } else {
                    prev.right = null;
                    cur = cur.right;
                }
            }
        }
        System.out.println(res);
    }

    public void inorderMorrisTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        TreeNode cur = root, prev;
        while (cur != null) {
            if (cur.left == null) {
                res.add(cur.val);
                cur = cur.right;
            } else {
                // find inorder prev
                prev = cur.left;
                while (prev.right != null && prev.right != cur) {
                    prev = prev.right;
                }
                if (prev.right == null) {
                    prev.right = cur;
                    cur = cur.left;
                } else {
                    res.add(cur.val);
                    prev.right = null;
                    cur = cur.right;
                }
            }
        }
        System.out.println(res);
    }

    public void postorderMorrisTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        TreeNode dump = new TreeNode(0);
        dump.left = root;
        TreeNode cur = dump, prev;
        while (cur != null) {
            if (cur.left == null) {
                cur = cur.right;
            } else {
                // find inorder prev
                prev = cur.left;
                while (prev.right != null && prev.right != cur) {
                    prev = prev.right;
                }
                if (prev.right == null) {
                    prev.right = cur;
                    cur = cur.left;
                } else {
                    TreeNode temp = cur.left;
                    int index = res.size();
                    while (temp != cur) {
                        res.add(index, temp.val);
                        temp = temp.right;
                    }
                    prev.right = null;
                    cur = cur.right;
                }
            }
        }
        System.out.println(res);
    }

    private TreeNode first = null, second = null, prev = new TreeNode(Integer.MIN_VALUE);

    public void recoverTree(TreeNode root) {
        inorderTraversal1(root);
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }

    public void inorderTraversal1(TreeNode root) {
        TreeNode cur = root, pre;
        while (cur != null) {
            if (cur.left == null) {
                if (prev.val >= cur.val) {
                    if (first == null) first = prev;
                    second = cur;
                }
                prev = cur;
                cur = cur.right;
            } else {
                // find prev
                pre = cur.left;
                while (pre.right != null && pre.right != cur) {
                    pre = pre.right;
                }
                if (pre.right == null) {
                    pre.right = cur;
                    cur = cur.left;
                } else {
                    pre.right = null;
                    if (prev.val >= cur.val) {
                        if (first == null) first = prev;
                        second = cur;
                    }
                    prev = cur;
                    cur = cur.right;
                }
            }
        }
    }

    private void recoverTreeHelper(TreeNode root) {
        if (root == null) return;
        recoverTreeHelper(root.left);
        if (prev.val >= root.val) {
            if (first == null) first = prev;
            second = root;
        }
        prev = root;
        recoverTreeHelper(root.right);
    }

    public boolean isValidBST2(TreeNode root) {
        if (root == null) return true;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null;
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (pre != null && root.val <= pre.val) return false;
            pre = root;
            root = root.right;
        }
        return true;
    }

    public boolean isValidBST(TreeNode root) {
        return isValidBSTHelper(root, null, null);
    }

    private boolean isValidBSTHelper(TreeNode root, TreeNode minNode, TreeNode maxNode) {
        if (root == null) return true;
        if (minNode != null && root.val <= minNode.val || maxNode != null && root.val >= maxNode.val) return false;
        return isValidBSTHelper(root.left, minNode, root) && isValidBSTHelper(root.right, root, maxNode);
    }

    public boolean isInterleave2(String s1, String s2, String s3) {
        if (s3.length() != s1.length() + s2.length()) return false;
        int m = s1.length(), n = s2.length();
        boolean[] dp = new boolean[n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[j] = true;
                } else if (i == 0) {
                    dp[j] = dp[j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1);
                } else if (j == 0) {
                    dp[j] = dp[j] && s1.charAt(i - 1) == s3.charAt(i + j - 1);
                } else {
                    dp[j] = (dp[j] && s1.charAt(i - 1) == s3.charAt(i + j - 1))
                            || (dp[j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
                }
            }
        }
        return dp[n];
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        if (s3.length() != s1.length() + s2.length()) return false;
        int m = s1.length(), n = s2.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = true;
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1);
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1);
                } else {
                    dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1))
                            || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
                }
            }
        }
        return dp[m][n];
    }

    public int numTrees2(int n) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        return numTreesHelper(map, 1, n);
    }

    private int numTreesHelper(Map<Integer, Integer> map, int start, int end) {
        if (start >= end) return 1;
        int k = end - start + 1;
        if (map.containsKey(k)) return map.get(k);
        int left, right, res = 0;
        for (int i = start; i <= end; i++) {
            left = numTreesHelper(map, start, i - 1);
            right = numTreesHelper(map, i + 1, end);
            res += left * right;
            map.put(i - start, left);
            map.put(end - i, right);
        }
        return res;
    }

    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i / 2; j++) {
                dp[i] += 2 * dp[j] * dp[i - j - 1];
            }
            dp[i] += (i % 2 != 0) ? dp[(i - 1) / 2] * dp[(i - 1) / 2] : 0;
        }
        return dp[n];
    }

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return new ArrayList<>();
        return generateTreesHelper(1, n);
    }

    private List<TreeNode> generateTreesHelper(int start, int end) {
        List<TreeNode> list = new ArrayList<>();
        if (start > end) {
            list.add(null);
            return list;
        }
        if (start == end) {
            list.add(new TreeNode(start));
            return list;
        }
        List<TreeNode> left, right;
        for (int i = start; i <= end; i++) {
            left = generateTreesHelper(start, i - 1);
            right = generateTreesHelper(i + 1, end);
            for (TreeNode lNode : left) {
                for (TreeNode rNode : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = lNode;
                    root.right = rNode;
                    list.add(root);
                }
            }
        }
        return list;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        inorderTraversalHelper(res, root);
        return res;
    }

    private void inorderTraversalHelper(List<Integer> res, TreeNode root) {
        if (root == null) return;
        inorderTraversalHelper(res, root.left);
        res.add(root.val);
        inorderTraversalHelper(res, root.right);
    }

    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        if (s == null || s.length() < 4 || s.length() > 12) return res;
        restoreIpAddressesHelper(res, s, "", 4);
        return res;
    }

    private void restoreIpAddressesHelper(List<String> res, String s, String str, int num) {
        if (num == 0) res.add(str.substring(1));
        int minI = Math.max(s.length() - 3 * (num - 1), 1), maxI = Math.min(s.length() - (num - 1), 3);
        for (int i = minI; i <= maxI; i++) {
            int n = Integer.parseInt(s.substring(0, i));
            if (i != 1 && !(n <= 255 && n >= (int) Math.pow(10, i - 1) && n < (int) Math.pow(10, i))) return;
            restoreIpAddressesHelper(res, s.substring(i), str + "." + s.substring(0, i), num - 1);
        }
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null || head.next == null) return head;
        ListNode root = new ListNode(0), p = root, q = head;
        root.next = head;
        while (true) {
            --m;
            --n;
            if (m <= 0 && n <= 0) break;
            if (m > 0) p = p.next;
            if (n > 0) q = q.next;
        }
        while (p.next != q) {
            ListNode t = p.next;
            p.next = t.next;
            t.next = q.next;
            q.next = t;
        }
        return root.next;
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
            if (one >= 1 && one <= 9) {
                dp[i] += dp[i - 1];
            }
            if (two >= 10 && two <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        subsetsWithDupHelper(res, new ArrayList<>(), nums, 0);
        return res;
    }

    private void subsetsWithDupHelper(List<List<Integer>> res, List<Integer> list, int[] nums, int index) {
        res.add(new ArrayList<>(list));
        for (int i = index; i < nums.length; i++) {
            if (i != index && nums[i] == nums[i - 1]) continue;
            list.add(nums[i]);
            subsetsWithDupHelper(res, list, nums, i + 1);
            list.remove(list.size() - 1);
        }
    }

    public List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        res.add(0);
        int i, j, k = 1;
        for (i = 1; i <= n; i++) {
            for (j = k - 1; j >= 0; j--) {
                res.add(res.get(j) + k);
            }
            k *= 2;
        }
        return res;
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while (i >= 0 || j >= 0) {
            int num1 = i >= 0 ? nums1[i] : Integer.MIN_VALUE, num2 = j >= 0 ? nums2[j] : Integer.MIN_VALUE;
            if (num1 >= num2) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
    }

    public boolean isScramble2(String s1, String s2) {
        int len = s1.length();
        boolean[][][] dp = new boolean[100][100][100];
        for (int i = len - 1; i >= 0; i--)
            for (int j = len - 1; j >= 0; j--) {
                dp[i][j][1] = (s1.charAt(i) == s2.charAt(j));
                for (int l = 2; i + l <= len && j + l <= len; l++) {
                    for (int n = 1; n < l; n++) {
                        dp[i][j][l] |= dp[i][j][n] && dp[i + n][j + n][l - n];
                        dp[i][j][l] |= dp[i][j + l - n][n] && dp[i + n][j][l - n];
                    }
                }
            }
        return dp[0][0][len];
    }

    public boolean isScramble(String s1, String s2) {
        if (s1.equals(s2)) return true;

        int[] letters = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            letters[s1.charAt(i) - 'a']++;
            letters[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) if (letters[i] != 0) return false;

        for (int i = 1; i < s1.length(); i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i))
                    && isScramble(s1.substring(i), s2.substring(i))) return true;
            if (isScramble(s1.substring(0, i), s2.substring(s2.length() - i))
                    && isScramble(s1.substring(i), s2.substring(0, s2.length() - i))) return true;
        }
        return false;
    }

    public ListNode partition(ListNode head, int x) {
        ListNode root = new ListNode(0), tail = root, cur = root, tag = null;
        root.next = head;
        while (tail.next != null) {
            tail = tail.next;
        }
        while (cur.next != tag && cur.next != tail) {
            if (cur.next.val >= x) {
                ListNode temp = cur.next;
                cur.next = temp.next;
                tail.next = temp;
                temp.next = null;
                tail = tail.next;
                if (tag == null) tag = temp;
            } else {
                cur = cur.next;
            }
        }
        return root.next;
    }

    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) return 0;
        int n = matrix[0].length, res = 0;
        int[] left = new int[n];
        int[] right = new int[n];
        int[] height = new int[n];
        Arrays.fill(right, n);
        for (char[] row : matrix) {
            int cur_left = 0, cur_right = n;
            for (int j = n - 1; j >= 0; j--) {
                if (row[j] == '1') right[j] = Math.min(right[j], cur_right);
                else {
                    right[j] = n;
                    cur_right = j;
                }
            }
            for (int j = 0; j < n; j++) {
                if (row[j] == '1') {
                    height[j]++;
                    left[j] = Math.max(left[j], cur_left);
                    res = Math.max(res, (right[j] - left[j]) * height[j]);
                } else {
                    height[j] = 0;
                    left[j] = 0;
                    cur_left = j + 1;
                }
            }
        }
        return res;
    }

    public int largestRectangleArea2(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int i = 0, maxArea = 0;
        int[] h = Arrays.copyOf(heights, heights.length + 1);
        while (i < h.length) {
            if (stack.isEmpty() || h[stack.peek()] <= h[i]) {
                stack.push(i++);
            } else {
                int t = stack.pop();
                maxArea = Math.max(maxArea, h[t] * (stack.isEmpty() ? i : i - stack.peek() - 1));
            }
        }
        return maxArea;
    }

    public int largestRectangleArea(int[] heights) {
        // Create an empty stack. The stack holds indexes of hist[] array
        // The bars stored in stack are always in increasing order of their
        // heights.
        Stack<Integer> s = new Stack<>();

        int max_area = 0; // Initialize max area
        int tp;  // To store top of stack
        int area_with_top; // To store area with top bar as the smallest bar

        // Run through all bars of given histogram
        int i = 0;
        while (i < heights.length) {
            // If this bar is higher than the bar on top stack, push it to stack
            if (s.empty() || heights[s.peek()] <= heights[i])
                s.push(i++);

                // If this bar is lower than top of stack, then calculate area of rectangle
                // with stack top as the smallest (or minimum height) bar. 'i' is
                // 'right index' for the top and element before top in stack is 'left index'
            else {
                tp = s.pop();  // store the top index

                // Calculate the area with hist[tp] stack as smallest bar
                area_with_top = heights[tp] * (s.empty() ? i : i - s.peek() - 1);

                // update max area, if needed
                if (max_area < area_with_top)
                    max_area = area_with_top;
            }
        }

        // Now pop the remaining bars from stack and calculate area with every
        // popped bar as the smallest bar
        while (!s.empty()) {
            tp = s.peek();
            s.pop();
            area_with_top = heights[tp] * (s.empty() ? i : i - s.peek() - 1);

            if (max_area < area_with_top)
                max_area = area_with_top;
        }

        return max_area;

    }

    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null) return null;
        ListNode p = head, q = p.next;
        while (q != null) {
            if (p.val != q.val) {
                p = q;
            } else {
                p.next = q.next;
            }
            q = q.next;
        }
        return head;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode p = root, q = root.next;
        while (q.next != null) {
            if (p.next.val != q.next.val) {
                if (p.next != q) {
                    p.next = q.next;
                } else {
                    p = p.next;
                }
                if (q.next.next == null) {
                    p = p.next;
                }
            }
            q = q.next;
        }
        p.next = null;
        return root.next;
    }

    public boolean search3(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1, mid;
        while (lo <= hi) {
            mid = (lo + hi) / 2;
            if (nums[mid] == target) return true;
            if (nums[lo] < nums[mid]) {
                //左边有序
                if (nums[lo] <= target && nums[mid] > target) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            } else if (nums[lo] > nums[mid]) {
                //右边有序
                if (nums[mid] < target && nums[hi] >= target) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            } else {
                if (nums[lo] == target) return true;
                lo++;
            }
        }
        return false;
    }

    public int removeDuplicates2(int[] nums) {
        if (nums.length < 2) return nums.length;
        int j = 2;
        for (int i = j; i < nums.length; i++) {
            if (nums[j - 1] == nums[j - 2] && nums[i] == nums[j - 1]) continue;
            nums[j++] = nums[i];
        }
        return j;
    }

    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0 || word.equals("")) return false;
        int m = board.length, n = board[0].length;
        boolean[] used = new boolean[m * n];
        Map<Character, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                List<Integer> list = map.getOrDefault(board[i][j], new ArrayList<>());
                list.add(i * n + j);
                map.put(board[i][j], list);
            }
        }
        List<Integer> list = map.getOrDefault(word.charAt(0), null);
        if (list == null) return false;
        int[] choice = {-n, n, -1, 1};
        for (int index : list) {
            used[index] = true;
            boolean b = existHelper(board, choice, used, word, m, n, index, 1);
            if (b) return true;
            used[index] = false;
        }
        return false;
    }

    private boolean existHelper(char[][] board, int[] choice, boolean[] used, String word, int m, int n, int index, int j) {
        if (j == word.length()) return true;
        for (int k, i = 0; i < choice.length; i++) {
            k = index + choice[i];
            if (k < 0 || k >= used.length || i > 1 && k / n != index / n || used[k] || board[k / n][k % n] != word.charAt(j))
                continue;
            used[k] = true;
            boolean b = existHelper(board, choice, used, word, m, n, k, j + 1);
            if (b) return true;
            used[k] = false;
        }
        return false;
    }

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        combineHelper(res, new ArrayList<>(), n, k, 1);
        return res;
    }

    private void combineHelper(List<List<Integer>> res, List<Integer> list, int n, int k, int index) {
        if (list.size() == k) {
            res.add(new ArrayList<>(list));
            return;
        }
        for (int i = index; i <= n - k + list.size() + 1; i++) {
            list.add(i);
            combineHelper(res, list, n, k, i + 1);
            list.remove(list.size() - 1);
        }
    }

    public String minWindow2(String s, String t) {
        if (s.length() == 0 || t.length() == 0) return "";
        Map<Character, Integer> dictT = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            int count = dictT.getOrDefault(t.charAt(i), 0);
            dictT.put(t.charAt(i), count + 1);
        }
        List<Pair<Integer, Character>> filteredS = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (dictT.containsKey(c)) {
                filteredS.add(new Pair<>(i, c));
            }
        }
        int required = dictT.size(), formed = 0;
        int left = 0, right = 0;
        Map<Character, Integer> windowS = new HashMap<>();
        int[] ans = {-1, 0, 0};
        while (right < filteredS.size()) {
            char c = filteredS.get(right).getValue();
            int count = windowS.getOrDefault(c, 0);
            windowS.put(c, count + 1);
            if (dictT.containsKey(c) && windowS.get(c).equals(dictT.get(c))) formed++;
            while (left <= right && formed == required) {
                c = filteredS.get(left).getValue();
                int end = filteredS.get(right).getKey();
                int start = filteredS.get(left).getKey();
                if (ans[0] == -1 || end - start + 1 < ans[0]) {
                    ans[0] = end - start + 1;
                    ans[1] = start;
                    ans[2] = end;
                }
                windowS.put(c, windowS.get(c) - 1);
                if (dictT.containsKey(c) && windowS.get(c) < dictT.get(c)) formed--;
                left++;
            }
            right++;
        }
        return ans[0] == -1 ? "" : s.substring(ans[1], ans[2] + 1);
    }

    public String minWindow(String s, String t) {
        if (s.length() == 0 || t.length() == 0) return "";
        Map<Character, Integer> dictT = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            int count = dictT.getOrDefault(t.charAt(i), 0);
            dictT.put(t.charAt(i), count + 1);
        }
        int required = dictT.size(), formed = 0;
        int left = 0, right = 0;
        Map<Character, Integer> windowS = new HashMap<>();
        int[] ans = {-1, 0, 0};
        while (right < s.length()) {
            char c = s.charAt(right);
            int count = windowS.getOrDefault(c, 0);
            windowS.put(c, count + 1);
            if (dictT.containsKey(c) && windowS.get(c).equals(dictT.get(c))) formed++;
            while (left <= right && formed == required) {
                c = s.charAt(left);
                if (ans[0] == -1 || right - left + 1 < ans[0]) {
                    ans[0] = right - left + 1;
                    ans[1] = left;
                    ans[2] = right;
                }
                windowS.put(c, windowS.get(c) - 1);
                if (dictT.containsKey(c) && windowS.get(c) < dictT.get(c)) formed--;
                left++;
            }
            right++;
        }
        return ans[0] == -1 ? "" : s.substring(ans[1], ans[2] + 1);
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

    public boolean searchMatrix2(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return false;
        int m = matrix.length, n = matrix[0].length, k = -1;
        int lo = 0, hi = m - 1, mid;
        while (lo <= hi) {
            mid = (lo + hi) / 2;
            if (matrix[mid][0] <= target && matrix[mid][n - 1] >= target) {
                k = mid;
                break;
            } else if (matrix[mid][n - 1] < target) lo = mid + 1;
            else hi = mid - 1;
        }
        if (k == -1) return false;
        lo = 0;
        hi = n - 1;
        while (lo <= hi) {
            mid = (lo + hi) / 2;
            if (matrix[k][mid] == target) return true;
            else if (matrix[k][mid] < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return false;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return false;
        int m = matrix.length, n = matrix[0].length, lo = 0, hi = m * n - 1, mid, i, j;
        while (lo <= hi) {
            mid = (lo + hi) / 2;
            i = mid / n;
            j = mid % n;
            if (matrix[i][j] == target) return true;
            else if (matrix[i][j] < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return false;
    }

    public void setZeroes2(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean isSet = false;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) isSet = true;
            for (int j = 0; j < n; j++) {
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
        if (isSet) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        Set<Integer> rowTags = new HashSet<>();
        Set<Integer> colTags = new HashSet<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    rowTags.add(i);
                    colTags.add(j);
                }
            }
        }
        for (int k : rowTags) {
            for (int j = 0; j < n; j++) {
                matrix[k][j] = 0;
            }
        }
        for (int k : colTags) {
            for (int i = 0; i < m; i++) {
                matrix[i][k] = 0;
            }
        }
    }

    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[i + 1][j + 1] = Math.min(Math.min(dp[i][j + 1], dp[i + 1][j]) + 1,
                        dp[i][j] + (word1.charAt(i) == word2.charAt(j) ? 0 : 1));    //删除 vs 插入 vs 替换或相同
            }
        }
        return dp[m][n];
    }

    public String simplifyPath(String path) {
        String[] strs = path.split("/");
        Stack<String> stack = new Stack<>();
        for (String str : strs) {
            switch (str) {
                case "":
                case ".":
                    break;
                case "..":
                    if (!stack.empty()) stack.pop();
                    break;
                default:
                    stack.push(str);
                    break;
            }
        }
        StringBuilder sb = new StringBuilder();
        if (stack.empty()) sb.append("/");
        while (!stack.empty()) {
            sb.insert(0, stack.pop());
            sb.insert(0, "/");
        }
        return sb.toString();
    }

    public int climbStairs(int n) {
        if (n == 0) return 0;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int mySqrt(int x) {
        if (x == 0) return 0;
        int left = 1, right = Integer.MAX_VALUE;
        while (true) {
            int mid = left + (right - left) / 2;
            if (mid > x / mid) {
                right = mid - 1;
            } else {
                if (mid + 1 > x / (mid + 1)) return mid;
                left = mid + 1;
            }
        }
    }

    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        int index = 0, sumWidth = 0;
        for (int i = 0; i < words.length; i++) {
            sumWidth += words[i].length();
            if (i < words.length - 1 && sumWidth + 1 + words[i + 1].length() <= maxWidth) {
                sumWidth += 1;
            } else if (i == words.length - 1) {
                StringBuilder sb = new StringBuilder();
                for (int j = index; j < i; j++) {
                    sb.append(words[j]);
                    sb.append(" ");
                }
                sb.append(words[i]);
                int k = maxWidth - sb.toString().length();
                for (int j = 0; j < k; j++) {
                    sb.append(" ");
                }
                res.add(sb.toString());
            } else {
                StringBuilder sb = new StringBuilder();
                int m = i - index, n = maxWidth;
                for (int j = index; j <= i; j++) {
                    n -= words[j].length();
                }
                for (int j = index; j < i; j++) {
                    sb.append(words[j]);
                    int t = (int) Math.ceil(1.0 * n / m);
                    for (int k = 0; k < t; k++) {
                        sb.append(" ");
                    }
                    m--;
                    n -= t;
                }
                sb.append(words[i]);
                int k = maxWidth - sb.toString().length();
                for (int j = 0; j < k; j++) {
                    sb.append(" ");
                }
                res.add(sb.toString());
                index = i + 1;
                sumWidth = 0;
            }
        }
        return res;
    }

    public String addBinary(String a, String b) {
        StringBuilder res = new StringBuilder();
        int i = a.length() - 1, j = b.length() - 1, m, n, c = 0;
        while (i >= 0 || j >= 0) {
            if (i >= 0) c += a.charAt(i) - '0';
            if (j >= 0) c += b.charAt(j) - '0';
            res.insert(0, c % 2);
            c /= 2;
            i--;
            j--;
        }
        if (c == 1) res.insert(0, c);
        return res.toString();
    }

    public int[] plusOne(int[] digits) {
        int c = 1;
        for (int i = digits.length - 1; i >= 0; i--) {
            digits[i] += c;
            if (digits[i] < 10) {
                return digits;
            } else {
                digits[i] %= 10;
                c = 1;
            }
        }
        int[] res = new int[digits.length + 1];
        res[0] = 1;
        return res;
    }

    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int n = grid.length, m = grid[0].length;
        int dp[] = new int[n];
        for (int i = 0; i < n; i++) {
            dp[i] = (i == 0 ? 0 : dp[i - 1]) + grid[i][0];
        }
        for (int j = 1; j < m; j++) {
            dp[0] += grid[0][j];
            for (int i = 1; i < n; i++) {
                dp[i] = Math.min(dp[i], dp[i - 1]) + grid[i][j];
            }
        }
        return dp[n - 1];
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) return 0;
        int n = obstacleGrid.length, m = obstacleGrid[0].length;
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            if (i != 0 && dp[i - 1] == 0) break;
            if (obstacleGrid[i][0] != 1) dp[i] = 1;
        }
        for (int j = 1; j < m; j++) {
            if (dp[0] == 1 && obstacleGrid[0][j] == 1) dp[0] = 0;
            for (int i = 1; i < n; i++) {
                dp[i] = obstacleGrid[i][j] == 1 ? 0 : dp[i] + dp[i - 1];
            }
        }
        return dp[n - 1];
    }

    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            dp[i] = 1;
        }
        for (int j = 1; j < m; j++) {
            for (int i = 1; i < n; i++) {
                dp[i] += dp[i - 1];
            }
        }
        return dp[n - 1];
    }

    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return head;
        int len = 1, sum = 0;
        ListNode p = head, q = head, r = head;
        while (r.next != null) {
            len++;
            r = r.next;
        }
        k %= len;
        while (q.next != null) {
            sum++;
            if (sum > k) {
                p = p.next;
            }
            q = q.next;
        }
        System.out.println(len + "\n" + k);
        System.out.println(p.val + "\n" + q.val);
        q.next = head;
        r = p.next;
        p.next = null;
        return r;
    }

    public String getPermutation(int n, int k) {
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            nums[i] = i + 1;
        }
        for (int i = 0; i < k - 1; i++) {
            nextPermutation(nums);
        }
        StringBuilder sb = new StringBuilder();
        for (int item : nums) {
            sb.append(item);
        }
        return sb.toString();
    }

    public int[][] generateMatrix(int n) {
        // Declaration
        int[][] matrix = new int[n][n];

        // Edge Case
        if (n == 0) {
            return matrix;
        }

        // Normal Case
        int rowStart = 0;
        int rowEnd = n - 1;
        int colStart = 0;
        int colEnd = n - 1;
        int num = 1; //change

        while (rowStart <= rowEnd && colStart <= colEnd) {
            for (int i = colStart; i <= colEnd; i++) {
                matrix[rowStart][i] = num++; //change
            }
            rowStart++;

            for (int i = rowStart; i <= rowEnd; i++) {
                matrix[i][colEnd] = num++; //change
            }
            colEnd--;

            for (int i = colEnd; i >= colStart; i--) {
                if (rowStart <= rowEnd)
                    matrix[rowEnd][i] = num++; //change
            }
            rowEnd--;

            for (int i = rowEnd; i >= rowStart; i--) {
                if (colStart <= colEnd)
                    matrix[i][colStart] = num++; //change
            }
            colStart++;
        }

        return matrix;
    }

    public int lengthOfLastWord(String s) {
        int res = 0, tag = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) != ' ') {
                res++;
                if (tag == 0) tag = 1;
            } else {
                if (tag == 1) break;
            }
        }
        return res;
    }

    public List<Interval> insert2(List<Interval> intervals, Interval newInterval) {
        List<Interval> result = new ArrayList<>();
        int i = 0;
        int start = newInterval.start;
        int end = newInterval.end;

        while (i < intervals.size() && intervals.get(i).end < start) {
            result.add(intervals.get(i++));
        }

        while (i < intervals.size() && intervals.get(i).start <= end) {
            start = Math.min(start, intervals.get(i).start);
            end = Math.max(end, intervals.get(i).end);
            i++;
        }
        result.add(new Interval(start, end));

        while (i < intervals.size()) result.add(intervals.get(i++));
        return result;
    }

    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        if (intervals == null || intervals.size() == 0) {
            intervals = new ArrayList<>();
            intervals.add(newInterval);
            return intervals;
        }
        if (newInterval == null) return intervals;
        int lo = findInterval(intervals, newInterval.start);
        int hi = findInterval(intervals, newInterval.end);
        List<Interval> res = new ArrayList<>();
        Interval interval = new Interval();
        for (int i = 0; i < intervals.size(); i++) {
            if (i == lo) {
                if (intervals.get(i).end >= newInterval.start) {
                    interval.start = Math.min(intervals.get(i).start, newInterval.start);
                } else {
                    interval.start = newInterval.start;
                }
                i = hi - 1;
            } else if (i == hi) {
                if (intervals.get(i).start <= newInterval.end) {
                    interval.start = Math.min(intervals.get(i).end, newInterval.end);
                } else {
                    interval.start = newInterval.end;
                }
                res.add(interval);
            } else {
                res.add(intervals.get(i));
            }
        }
        return res;
    }

    private int findInterval(List<Interval> intervals, int val) {
        int lo = 0, hi = intervals.size() - 1, mid = 0;
        while (lo <= hi) {
            mid = (lo + hi) / 2;
            if (intervals.get(mid).start <= val && intervals.get(mid).end >= val) {
                break;
            } else if (intervals.get(mid).start > val) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        return mid;
    }

    public List<Interval> merge(List<Interval> intervals) {
        if (intervals == null || intervals.size() <= 1) return intervals;
        intervals.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return Integer.compare(o1.start, o2.start);
            }
        });
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

    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0) return res;
        int i, j, m = matrix.length, n = matrix[0].length, l = Math.min(m, n) / 2;
        for (i = 0; i < l; i++) {
            for (j = i; j < n - i - 1; j++) res.add(matrix[i][j]);
            for (j = i; j < m - i - 1; j++) res.add(matrix[j][n - i - 1]);
            for (j = i; j < n - i - 1; j++) res.add(matrix[m - i - 1][n - j - 1]);
            for (j = i; j < m - i - 1; j++) res.add(matrix[m - j - 1][i]);
        }
        if (m <= n && m % 2 == 1)
            for (j = i; j <= n - i - 1; j++) res.add(matrix[i][j]);
        if (m > n && n % 2 == 1)
            for (j = i; j <= m - i - 1; j++) res.add(matrix[j][i]);
        return res;
    }

    public int maxSubArray(int[] nums) {
        int cur = nums[0], res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            cur = Math.max(cur + nums[i], nums[i]);
            res = Math.max(cur, res);
        }
        return res;
    }

    public int totalNQueens(int n) {
        return solveNQueens(n).size();
    }

    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                board[i][j] = '.';
        solveNQueensDFS(res, board, 0);
        return res;
    }

    private void solveNQueensDFS(List<List<String>> res, char[][] board, int colIndex) {
        if (colIndex == board.length) {
            res.add(solveNQueensConstruct(board));
            return;
        }
        for (int i = 0; i < board.length; i++) {
            if (!solveNQueensValid(board, i, colIndex)) continue;
            board[i][colIndex] = 'Q';
            solveNQueensDFS(res, board, colIndex + 1);
            board[i][colIndex] = '.';
        }
    }

    private boolean solveNQueensValid(char[][] board, int x, int y) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < y; j++) {
                if (board[i][j] == 'Q' && (x + j == y + i || x + y == i + j || x == i))
                    return false;
            }
        }
        return true;
    }

    private List<String> solveNQueensConstruct(char[][] board) {
        List<String> res = new ArrayList<>();
        for (char[] item : board) {
            String str = new String(item);
            res.add(str);
        }
        return res;
    }

    public double myPow(double x, int n) {
        if (n == 0) return 1;
        if (n < 0) {
            x = 1 / x;
            if (n == Integer.MIN_VALUE) {
                return x * myPow(x, Integer.MAX_VALUE);
            }
            n = -n;
        }
        return ((n % 2 == 0) ? 1 : x) * myPow(x * x, n / 2);
    }

    public List<List<String>> groupAnagrams2(String[] strs) {
        int[] prime = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103};//最多10609个z
        List<List<String>> res = new ArrayList<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (String str : strs) {
            int key = 1;
            for (char c : str.toCharArray()) {
                key *= prime[c - '0'];
            }
            List<String> list;
            if (map.containsKey(key)) {
                list = res.get(map.get(key));
            } else {
                list = new ArrayList<>();
                res.add(list);
                map.put(key, res.size() - 1);
            }
            list.add(str);
        }
        return res;
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> res = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        int index = 0;
        for (String item : strs) {
            String str = getSortedStr(item);
            int k = map.getOrDefault(str, -1);
            if (k == -1) {
                map.put(str, index++);
                List<String> list = new ArrayList<>();
                list.add(item);
                res.add(list);
            } else {
                List<String> list = res.get(k);
                list.add(item);
            }
        }
        return res;
    }

    private String getSortedStr(String str) {
        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        return String.valueOf(chars);
    }

    public void rotate(int[][] matrix) {
        int n = matrix.length, m = n / 2, temp;
        for (int i = 0; i < m; i++) {
            for (int j = i; j < n - i - 1; j++) {
                temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        permuteUniqueHelper(res, nums, 0);
        return res;
    }

    private void permuteUniqueHelper(List<List<Integer>> res, int[] nums, int index) {
        if (index == nums.length) {
            List<Integer> list = new ArrayList<>();
            for (Integer item : nums) list.add(item);
            res.add(list);
            return;
        }
        for (int i = index; i < nums.length; i++) {
            //当我们枚举第i个位置的元素时，若要把后面第j个元素和i交换，则先要保证[i…j-1]范围内没有和位置j相同的元素。
            if (i != index && findTarget(nums, index, i, nums[i])) continue;
            swap(nums, i, index);
            permuteUniqueHelper(res, nums, index + 1);
            swap(nums, i, index);
        }
    }

    private boolean findTarget(int[] nums, int start, int end, int target) {
        for (int i = start; i < end; i++) if (nums[i] == target) return true;
        return false;
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

    public int jump2(int[] nums) {
        int cur = 0, last = 0, res = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > last) {
                res++;
                last = cur;
            }
            cur = Math.max(cur, i + nums[i]);
        }
        return res;
    }

    public int jump(int[] nums) {
        int i = 0, res = 0, cur = 0, pre;
        while (cur < nums.length - 1) {
            res++;
            for (pre = cur; i <= pre; i++) {
                cur = Math.max(cur, i + nums[i]);
            }
        }
        return res;
    }

    public boolean isMatch1(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = i > 0 && dp[i - 1][j] || dp[i][j - 1];
                } else {
                    dp[i][j] = i > 0 && dp[i - 1][j - 1] && (p.charAt(j - 1) == '?' || s.charAt(i - 1) == p.charAt(j - 1));
                }
            }
        }
        return dp[m][n];
    }

    public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] pos = new int[m + n];

        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int p1 = i + j, p2 = p1 + 1;
                int sum = mul + pos[p2];
                pos[p1] += sum / 10;
                pos[p2] = sum % 10;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int item : pos) if (sb.length() != 0 || item != 0) sb.append(item);
        return sb.length() == 0 ? "0" : sb.toString();
    }

    public int trap(int[] height) {
        if (height == null || height.length == 0) return 0;
        int i = 0, j = height.length - 1, m = height[i], n = height[j], sum = 0, res = 0;
        while (i <= j) {
            if (height[i] < m) {
                sum += m - height[i];
            } else {
                m = height[i];
                res += sum;
                sum = 0;
            }
            if (height[j] < n) {
                sum += n - height[j];
            } else {
                n = height[j];
                res += sum;
                sum = 0;
            }
            if (height[i] <= height[j]) {
                i++;
            } else {
                j--;
            }
        }
        return res;
    }

    public int firstMissingPositive2(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            if (nums[i] == i + 1 || nums[i] <= 0 || nums[i] > nums.length) i++;
            else if (nums[nums[i] - 1] != nums[i]) swap(nums, i, nums[i] - 1);
            else i++;
        }
        i = 0;
        while (i < nums.length && nums[i] == i + 1) i++;
        return i + 1;
    }

    public int firstMissingPositive(int[] nums) {
        Arrays.sort(nums);
        int num = 1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= 0 || (i > 0 && nums[i] == nums[i - 1])) continue;
            if (nums[i] == num) {
                num++;
            } else {
                break;
            }
        }
        return num;
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        combinationSumHelper2(res, new ArrayList<>(), candidates, target, 0);
        return res;
    }

    private void combinationSumHelper2(List<List<Integer>> res, List<Integer> list, int[] candidates, int remain, int index) {
        if (remain < 0) return;
        if (remain == 0) {
            res.add(new ArrayList<>(list));
            return;
        }
        for (int i = index; i < candidates.length; i++) {
            if (i > index && candidates[i] == candidates[i - 1] && list.size() == 0) continue;
            list.add(candidates[i]);
            combinationSumHelper2(res, list, candidates, remain - candidates[i], i + 1);
            list.remove(list.size() - 1);
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        combinationSumHelper(res, new ArrayList<>(), candidates, target, 0, 0);
        return res;
    }

    private void combinationSumHelper(List<List<Integer>> res, List<Integer> list, int[] candidates, int target, int sum, int index) {
        if (sum == target) {
            res.add(new ArrayList<>(list));
            return;
        }
        if (sum > target) return;
        for (int i = index; i < candidates.length; i++) {
            list.add(candidates[i]);
            combinationSumHelper(res, list, candidates, target, sum + candidates[i], i);
            list.remove(list.size() - 1);
        }
    }

    public void solveSudoku(char[][] board) {
        if (board == null || board.length == 0)
            return;
        solve(board);
    }

    private boolean solve(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {//trial. Try 1 through 9
                        if (isValid(board, i, j, c)) {
                            board[i][j] = c; //Put c for this cell
                            if (solve(board))
                                return true; //If it's the solution return true
                            else
                                board[i][j] = '.'; //Otherwise go back
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    private boolean isValid(char[][] board, int row, int col, char c) {
        for (int i = 0; i < 9; i++) {
            if (board[i][col] != '.' && board[i][col] == c) return false; //check row
            if (board[row][i] != '.' && board[row][i] == c) return false; //check column
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] != '.' &&
                    board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) return false; //check 3*3 block
        }
        return true;
    }

    public int searchInsert(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1, mid;
        while (lo <= hi) {
            mid = (lo + hi) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    public int[] searchRange(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1, mid = 0, start = 0, end = nums.length - 1;
        while (lo <= hi) {
            mid = (lo + hi) / 2;
            if (nums[mid] == target) {
                break;
            } else if (nums[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (lo > hi) return new int[]{-1, -1};

        lo = 0;
        hi = mid - 1 > 0 ? mid - 1 : 0;
        while (lo <= hi) {
            start = (lo + hi) / 2;
            if (nums[start] != target && nums[start + 1] == target) {
                break;
            } else if (nums[start] == target) {
                hi = start - 1;
            } else {
                lo = start + 1;
            }
        }
        lo = mid + 1 < nums.length - 1 ? mid + 1 : nums.length - 1;
        hi = nums.length - 1;
        while (lo <= hi) {
            end = (lo + hi) / 2;
            if (nums[end] != target && nums[end - 1] == target) {
                break;
            } else if (nums[end] == target) {
                lo = end + 1;
            } else {
                hi = end - 1;
            }
        }
        return new int[]{nums[start] == target ? start : start + 1, nums[end] == target ? end : end - 1};
    }

    public int search2(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1, mid;
        while (lo <= hi) {
            mid = (lo + hi) / 2;
            if (target == nums[mid]) return mid;
            if (nums[lo] <= nums[mid]) {    //左半段有序
                //target在这段里
                if (target >= nums[lo] && target < nums[hi]) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            } else {    //右半段有序
                //target在这段里
                if (target > nums[mid] && target <= nums[hi]) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        return -1;
    }

    //上面那个好理解
    public int search(int[] nums, int target) {
        int i = 0, j = nums.length - 1, k;
        while (i <= j) {
            k = (i + j) / 2;
            if (nums[k] < target) {
                if (nums[i] > nums[j] && nums[i] >= nums[k] && nums[i] <= target) {
                    j = k - 1;
                } else {
                    i = k + 1;
                }
            } else if (nums[k] > target) {
                if (nums[i] > nums[j] && nums[i] <= nums[k] && nums[i] > target) {
                    i = k + 1;
                } else {
                    j = k - 1;
                }
            } else {
                return k;
            }
        }
        return -1;
    }

    public int longestValidParentheses(String s) {
        int[] dp = new int[s.length()];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (!stack.empty()) {
                    int j = stack.pop();
                    dp[i] = 1;
                    dp[j] = 1;
                }
            }
        }
        int num = 0, res = 0;
        for (int item : dp) {
            if (item == 0) {
                res = Math.max(num, res);
                num = 0;
            } else {
                num++;
            }
        }
        return Math.max(num, res);
    }

    public void nextPermutation(int[] nums) {
        int l = Integer.MIN_VALUE, k = findK(nums);
        if (k == -1) {
            reverse(nums, 0);
            return;
        }
        for (int i = k + 1; i < nums.length; i++) {
            if (nums[i] > nums[k] && (l == Integer.MIN_VALUE || nums[i] <= nums[l])) l = i;
        }
        swap(nums, k, l);
        reverse(nums, k + 1);
    }

    private int findK(int[] nums) {
        int k = nums.length - 2;
        while (k >= 0 && nums[k] >= nums[k + 1]) k--;
        return k;
    }

    private void reverse(int[] nums, int from) {
        int i = from, j = nums.length - 1, t;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    public List<Integer> findSubstring2(String s, String[] words) {
        List<Integer> res = new ArrayList<Integer>();
        int wordNum = words.length;
        if (wordNum == 0) {
            return res;
        }
        int wordLen = words[0].length();
        HashMap<String, Integer> allWords = new HashMap<String, Integer>();
        for (String w : words) {
            int value = allWords.getOrDefault(w, 0);
            allWords.put(w, value + 1);
        }
        //将所有移动分成 wordLen 类情况
        for (int j = 0; j < wordLen; j++) {
            HashMap<String, Integer> hasWords = new HashMap<String, Integer>();
            int num = 0; //记录当前 HashMap2（这里的 hasWords 变量）中有多少个单词
            //每次移动一个单词长度
            for (int i = j; i < s.length() - wordNum * wordLen + 1; i = i + wordLen) {
                boolean hasRemoved = false; //防止情况三移除后，情况一继续移除
                while (num < wordNum) {
                    String word = s.substring(i + num * wordLen, i + (num + 1) * wordLen);
                    if (allWords.containsKey(word)) {
                        int value = hasWords.getOrDefault(word, 0);
                        hasWords.put(word, value + 1);
                        //出现情况三，遇到了符合的单词，但是次数超了
                        if (hasWords.get(word) > allWords.get(word)) {
                            // hasWords.put(word, value);
                            hasRemoved = true;
                            int removeNum = 0;
                            //一直移除单词，直到次数符合了
                            while (hasWords.get(word) > allWords.get(word)) {
                                String firstWord = s.substring(i + removeNum * wordLen, i + (removeNum + 1) * wordLen);
                                int v = hasWords.get(firstWord);
                                hasWords.put(firstWord, v - 1);
                                removeNum++;
                            }
                            num = num - removeNum + 1; //加 1 是因为我们把当前单词加入到了 HashMap 2 中
                            i = i + (removeNum - 1) * wordLen; //这里依旧是考虑到了最外层的 for 循环，看情况二的解释
                            break;
                        }
                        //出现情况二，遇到了不匹配的单词，直接将 i 移动到该单词的后边（但其实这里
                        //只是移动到了出现问题单词的地方，因为最外层有 for 循环， i 还会移动一个单词
                        //然后刚好就移动到了单词后边）
                    } else {
                        hasWords.clear();
                        i = i + num * wordLen;
                        num = 0;
                        break;
                    }
                    num++;
                }
                if (num == wordNum) {
                    res.add(i);

                }
                //出现情况一，子串完全匹配，我们将上一个子串的第一个单词从 HashMap2 中移除
                if (num > 0 && !hasRemoved) {
                    String firstWord = s.substring(i, i + wordLen);
                    int v = hasWords.get(firstWord);
                    hasWords.put(firstWord, v - 1);
                    num = num - 1;
                }

            }

        }
        return res;
    }

    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        int wordNum = words.length;
        if (wordNum == 0) return res;
        int wordLen = words[0].length();
        // allWords 存放所有单词
        Map<String, Integer> allWords = new HashMap<>();
        for (String word : words) {
            int value = allWords.getOrDefault(word, 0);
            allWords.put(word, value + 1);
        }
        // 遍历所有子串
        for (int i = 0; i < s.length() - wordNum * wordLen + 1; i++) {
            // 存含有的单词
            Map<String, Integer> hasWords = new HashMap<>();
            int num = 0;
            while (num < wordNum) {
                String word = s.substring(i + num * wordLen, i + (num + 1) * wordLen);
                if (allWords.containsKey(word)) {
                    int value = hasWords.getOrDefault(word, 0);
                    hasWords.put(word, value + 1);
                    if (hasWords.get(word) > allWords.get(word)) break;
                } else break;
                num++;
            }
            if (num == wordNum) res.add(i);
        }
        return res;
    }

    public int strStr(String haystack, String needle) {
        return haystack.indexOf(needle);
    }

    public int removeElement(int[] nums, int val) {
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != val) {
                nums[j] = nums[i];
                j++;
            }
        }
        return j;
    }

    public int removeDuplicates(int[] nums) {
        int j = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[j - 1]) {
                nums[j] = nums[i];
                j++;
            }
        }
        return j;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k == 1) return head;
        ListNode root = new ListNode(0), p = root, q = root;
        root.next = head;
        int num = 0;
        while (q.next != null) {
            q = q.next;
            num++;
            if (num == k) {
                num = 0;
                q = reverseHelper(p, q);
                p = q;
            }
        }
        return root.next;
    }

    private ListNode reverseHelper(ListNode p, ListNode q) {
        ListNode res = p.next;
        ListNode t = p.next.next, x = t.next;
        p.next.next = q.next;
        while (t != q) {
            t.next = p.next;
            p.next = t;
            t = x;
            x = t.next;
        }
        q.next = p.next;
        p.next = q;
        return res;
    }

    public ListNode swapPairs(ListNode head) {
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode p = root, q = head;
        while (q != null && q.next != null) {
            p.next = q.next;
            q.next = q.next.next;
            p.next.next = q;
            p = q;
            q = q.next;
        }
        return root.next;
    }

    public ListNode mergeKLists2(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return Integer.compare(o1.val, o2.val);
            }
        });
        ListNode head = new ListNode(0), tail = head;
        for (ListNode node : lists)
            if (node != null) queue.add(node);
        while (queue.size() != 0) {
            tail.next = queue.poll();
            tail = tail.next;
            if (tail.next != null) queue.add(tail.next);
        }
        return head.next;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        int index = -1;
        for (int i = 0; i < lists.length; i++) {
            if (lists[i] == null) continue;
            if (index == -1 || lists[index].val > lists[i].val) index = i;
        }
        if (index == -1) return null;
        ListNode l = lists[index];
        lists[index] = lists[index].next;
        l.next = mergeKLists(lists);
        return l;
    }

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        generateParenthesisHelper(res, "", n, 0, 0);
        return res;
    }

    private void generateParenthesisHelper(List<String> res, String str, int n, int a, int b) {
        if (a == n && b == n) {
            res.add(str);
            return;
        }
        if (a < n) {
            generateParenthesisHelper(res, str + "(", n, a + 1, b);
        }
        if (b < a) {
            generateParenthesisHelper(res, str + ")", n, a, b + 1);
        }
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode l = new ListNode(0), p = l;
        while (l1 != null || l2 != null) {
            if (l1 == null || (l2 != null && l1.val > l2.val)) {
                l.next = l2;
                l2 = l2.next;
            } else {
                l.next = l1;
                l1 = l1.next;
            }
            l = l.next;
        }
        return p.next;
    }

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        char c, ch;
        for (int i = 0; i < s.length(); i++) {
            c = s.charAt(i);
            switch (c) {
                case '(':
                case '[':
                case '{':
                    stack.push(c);
                    break;
                case ')':
                case ']':
                case '}':
                    if (stack.empty()) return false;
                    ch = stack.pop();
                    if (c == ')' && ch != '(' || c == ']' && ch != '[' || c == '}' && ch != '{') return false;
                    break;
            }
        }
        return stack.empty();
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode res = new ListNode(0);
        res.next = head;
        ListNode pre = res, p = res;
        while (p.next != null) {
            if (n < 0) {
                pre = pre.next;
            }
            p = p.next;
            n--;
        }
        pre.next = pre.next.next;
        return res.next;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        ArrayList<List<Integer>> res = new ArrayList<List<Integer>>();
        int len = nums.length;
        if (nums == null || len < 4)
            return res;

        Arrays.sort(nums);

        int max = nums[len - 1];
        if (4 * nums[0] > target || 4 * max < target)
            return res;

        int i, z;
        for (i = 0; i < len; i++) {
            z = nums[i];
            if (i > 0 && z == nums[i - 1])// avoid duplicate
                continue;
            if (z + 3 * max < target) // z is too small
                continue;
            if (4 * z > target) // z is too large
                break;
            if (4 * z == target) { // z is the boundary
                if (i + 3 < len && nums[i + 3] == z)
                    res.add(Arrays.asList(z, z, z, z));
                break;
            }

            threeSumForFourSum(nums, target - z, i + 1, len - 1, res, z);
        }

        return res;
    }

    /*
     * Find all possible distinguished three numbers adding up to the target
     * in sorted array nums[] between indices low and high. If there are,
     * add all of them into the ArrayList fourSumList, using
     * fourSumList.add(Arrays.asList(z1, the three numbers))
     */
    public void threeSumForFourSum(int[] nums, int target, int low, int high, ArrayList<List<Integer>> fourSumList,
                                   int z1) {
        if (low + 1 >= high)
            return;

        int max = nums[high];
        if (3 * nums[low] > target || 3 * max < target)
            return;

        int i, z;
        for (i = low; i < high - 1; i++) {
            z = nums[i];
            if (i > low && z == nums[i - 1]) // avoid duplicate
                continue;
            if (z + 2 * max < target) // z is too small
                continue;

            if (3 * z > target) // z is too large
                break;

            if (3 * z == target) { // z is the boundary
                if (i + 1 < high && nums[i + 2] == z)
                    fourSumList.add(Arrays.asList(z1, z, z, z));
                break;
            }

            twoSumForFourSum(nums, target - z, i + 1, high, fourSumList, z1, z);
        }

    }

    /*
     * Find all possible distinguished two numbers adding up to the target
     * in sorted array nums[] between indices low and high. If there are,
     * add all of them into the ArrayList fourSumList, using
     * fourSumList.add(Arrays.asList(z1, z2, the two numbers))
     */
    public void twoSumForFourSum(int[] nums, int target, int low, int high, ArrayList<List<Integer>> fourSumList,
                                 int z1, int z2) {

        if (low >= high)
            return;

        if (2 * nums[low] > target || 2 * nums[high] < target)
            return;

        int i = low, j = high, sum, x;
        while (i < j) {
            sum = nums[i] + nums[j];
            if (sum == target) {
                fourSumList.add(Arrays.asList(z1, z2, nums[i], nums[j]));

                x = nums[i];
                while (++i < j && x == nums[i]) // avoid duplicate
                    ;
                x = nums[j];
                while (i < --j && x == nums[j]) // avoid duplicate
                    ;
            }
            if (sum < target)
                i++;
            if (sum > target)
                j--;
        }
        return;
    }

    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits == null || digits.equals("")) {
            return res;
        }
        String[] s = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        letterHelper(res, s, digits, 0, "");
        return res;
    }

    private void letterHelper(List<String> res, String[] s, String digits, int index, String str) {
        if (index == digits.length()) {
            res.add(str);
            return;
        }
        int k = digits.charAt(index) - '0';
        for (int i = 0; i < s[k].length(); i++) {
            letterHelper(res, s, digits, index + 1, str + s[k].charAt(i));
        }
    }

    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length - 2; i++) {
            if (i != 0 && nums[i] == nums[i - 1]) continue;
            int j = i + 1, k = nums.length - 1, t;
            while (j < k) {
                t = nums[i] + nums[j] + nums[k];
                if (t < target) {
                    j++;
                } else {
                    k--;
                }
                if (res == Integer.MAX_VALUE || Math.abs(target - res) > Math.abs(target - t)) {
                    res = t;
                }
            }
        }
        return res;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {
            if (nums[i] > 0) break;
            if (i != 0 && nums[i] == nums[i - 1]) continue;
            int j = i + 1, k = nums.length - 1, t;
            while (j < k) {
                t = nums[i] + nums[j] + nums[k];
                if (t == 0) {
                    res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    while (j < k && nums[j] == nums[j + 1]) j++;
                    while (j < k && nums[k] == nums[k - 1]) k--;
                    j++;
                    k--;
                } else if (t < 0) j++;
                else k--;
            }
        }
        return res;
    }

    public int maxArea(int[] height) {
        int i = 0, j = height.length - 1, max = 0;
        while (i < j) {
            max = Math.max(max, Math.min(height[i], height[j]) * (j - i));
            if (height[i] <= height[j]) {
                i++;
            } else {
                j--;
            }
        }
        return max;
    }

    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2] || (i > 0 && (s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.')
                            && dp[i - 1][j]);
                } else {
                    dp[i][j] = i > 0 && dp[i - 1][j - 1] && (s.charAt(i - 1) == s.charAt(j - 1) || p.charAt(j - 1) == '.');
                }
            }
        }
        return dp[m][n];
    }

    public boolean isPalindrome1(int x) {
        if (x < 0 || (x != 0 && x % 10 == 0)) return false;
        int rev = 0;
        while (x > rev) {
            rev = rev * 10 + x % 10;
            x /= 10;
        }
        return (x == rev || x == rev / 10);
    }

    public boolean isPalindrome(int x) {
        String s = String.valueOf(x);
        int i = 0, j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                break;
            }
            i++;
            j--;
        }
        return i >= j;
    }

    public String longestPalindrome(String s) {
        if (s.length() == 0) {
            return "";
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = lengthOfPoint(s, i, i);
            int len2 = lengthOfPoint(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int lengthOfPoint(String s, int i, int j) {
        int i1 = i, j1 = j;
        while (i1 >= 0 && j1 < s.length() && s.charAt(i1) == s.charAt(j1)) {
            i1--;
            j1++;
        }
        return j1 - i1 - 1;
    }

}

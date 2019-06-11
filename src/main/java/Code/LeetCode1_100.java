package Code;

import Data.*;

import java.util.*;

/*
LeetCode 1-100
 */
public class LeetCode1_100 {

    public int longestValidParentheses2(String s) {
        int n = s.length(), max = 0;
        int[] dp = new int[n];
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(c);
                dp[i] = max;
            } else {
                if (!stack.empty()) {
                    stack.pop();
                    max += 2;
                    dp[i] = max;
                }
            }
        }
        return dp[n - 1];
    }

    public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        char[] ch = s.toCharArray();
        for (int i = 0; i < ch.length; i++) {
            if (ch[i] == '(') {
                stack.push(i);
            } else if (!stack.empty()) {
                ch[i] = '-';
                ch[stack.pop()] = '-';
            }
        }
        int max = 0, cur = 0;
        for (char c : ch) {
            if (c == '-') {
                cur++;
            } else {
                max = max > cur ? max : cur;
                cur = 0;
            }
        }
        max = max > cur ? max : cur;
        return max;
    }

    public List<Integer> findSubstring(String s, String[] words) {
        Set<Integer> result = new HashSet<>();
        if (s == null || s.length() == 0 || words == null || words.length == 0) {
            return new ArrayList<>(result);
        }
        int[] nums = new int[words.length];
        for (int i = 0; i < nums.length; i++) {
            nums[i] = i;
        }
        List<List<Integer>> lists = permuteUnique2(nums);
        for (List<Integer> list : lists) {
            String str = "";
            for (Integer i : list) {
                str += words[i];
            }
            int from, k = 0;
            while (true) {
                from = k;
                k = s.indexOf(str, from);
                if (k == -1) {
                    break;
                }
                result.add(k);
                k++;
            }
        }
        return new ArrayList<>(result);
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0), p = root, q = root, t;
        root.next = head;
        int n = 1;
        while (q.next != null) {
            if (n < k) {
                n++;
                q = q.next;
            } else {
                t = p.next;
                while (n != 1) {
                    n--;
                    q = p.next;
                    p.next = t.next;
                    t.next = t.next.next;
                    p.next.next = q;
                }
                p = q = t;
            }
        }
        return root.next;
    }

    public ListNode mergeKLists2(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return Integer.compare(o1.val, o2.val);
            }
        });
        ListNode head = new ListNode(0), tail = head;
        for (ListNode listNode : lists) {
            if (listNode != null) {
                queue.add(listNode);
            }
        }
        while (queue.size() != 0) {
            tail.next = queue.poll();
            tail = tail.next;
            if (tail.next != null) {
                queue.add(tail.next);
            }
        }
        return head.next;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        ListNode head = new ListNode(0), p = new ListNode(0), q = new ListNode(0);
        head.next = lists[0];
        for (int i = 1; i < lists.length; i++) {
            p = head;
            q = lists[i];
            while (p.next != null || q != null) {
                if (p.next == null) {
                    p.next = q;
                    break;
                } else if (q == null) {
                    break;
                } else {
                    if (p.next.val > q.val) {
                        ListNode t = new ListNode(q.val);
                        t.next = p.next;
                        p.next = t;
                        q = q.next;
                    }
                    p = p.next;
                }
            }
        }
        return head.next;
    }

    public boolean isMatch4(String s, String p) {
        int m = s.length(), n = p.length();
        int i = 0, j = 0, asterisk = -1, match = 0;
        while (i < m) {
            if (j < n && p.charAt(j) == '*') {
                match = i;
                asterisk = j++;
            } else if (j < n && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?')) {
                i++;
                j++;
            } else if (asterisk >= 0) {
                i = ++match;
                j = asterisk + 1;
            } else return false;
        }
        while (j < n && p.charAt(j) == '*') j++;
        return j == n;
    }

    public boolean isMatch3(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = i > 0 && dp[i - 1][j] || dp[i][j - 1];
                } else {
                    dp[i][j] = i > 0 && dp[i - 1][j - 1] && (p.charAt(j - 1) == '?' || p.charAt(j - 1) == s.charAt(i - 1));
                }
            }
        }
        return dp[m][n];
    }

    public boolean isMatch2(String s, String p) {
        if (p.isEmpty()) return s.isEmpty();
        if (p.length() == 1) return (s.length() == 1) && (p.charAt(0) == '.' || p.charAt(0) == s.charAt(0));
        if (p.charAt(1) != '*') {
            if (s.isEmpty()) return false;
            return (p.charAt(0) == '.' || p.charAt(0) == s.charAt(0)) && isMatch2(s.substring(1), p.substring(1));
        }
        while (!s.isEmpty() && (p.charAt(0) == '.' || p.charAt(0) == s.charAt(0))) {
            if (isMatch2(s, p.substring(2))) return true;
            s = s.substring(1);
        }
        return isMatch2(s, p.substring(2));
    }

    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2] || (i > 0 && (s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.') && dp[i - 1][j]);
                } else {
                    dp[i][j] = i > 0 && dp[i - 1][j - 1] && (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.');
                }
            }
        }
        return dp[m][n];
    }

    public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
        int[] A, B;
        if (nums1.length < nums2.length) {
            A = nums1;
            B = nums2;
        } else {
            A = nums2;
            B = nums1;
        }
        if (B.length == 0) {
            return 0;
        }
        int i, j, m = A.length, n = B.length, lmax = 0, rmax = 0, imin = 0, imax = m, imid = (m + n + 1) / 2;
        while (imin <= imax) {
            i = (imin + imax) / 2;
            j = imid - i;
            if (i < m && B[j - 1] > A[i]) {
                imin = i + 1;
            } else if (i > 0 && A[i - 1] > B[j]) {
                imax = i - 1;
            } else {
                if (i == 0) {
                    lmax = B[j - 1];
                } else if (j == 0) {
                    lmax = A[i - 1];
                } else {
                    lmax = Math.max(A[i - 1], B[j - 1]);
                }
                if ((m + n) % 2 == 1) {
                    return lmax;
                }
                if (i == m) {
                    rmax = B[j];
                } else if (j == n) {
                    rmax = A[i];
                } else {
                    rmax = Math.min(A[i], B[j]);
                }
                break;
            }
        }
        return (lmax + rmax) / 2.0;
    }

    public String getPermutation2(int n, int k) {
        int pos = 0;
        List<Integer> numbers = new ArrayList<>();
        int[] factorial = new int[n + 1];
        StringBuilder sb = new StringBuilder();

        // create an array of factorial lookup
        int sum = 1;
        factorial[0] = 1;
        for (int i = 1; i <= n; i++) {
            sum *= i;
            factorial[i] = sum;
        }
        // factorial[] = {1, 1, 2, 6, 24, ... n!}

        // create a list of numbers to get indices
        for (int i = 1; i <= n; i++) {
            numbers.add(i);
        }
        // numbers = {1, 2, 3, 4}

        k--;

        for (int i = 1; i <= n; i++) {
            int index = k / factorial[n - i];
            sb.append(String.valueOf(numbers.get(index)));
            numbers.remove(index);
            k -= index * factorial[n - i];
        }

        return String.valueOf(sb);
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[] cur = new int[m];
        cur[0] = obstacleGrid[0][0] == 1 ? 0 : 1;
        for (int i = 1; i < m; i++) {
            if (cur[i - 1] != 0 && obstacleGrid[i][0] != 1) {
                cur[i] = 1;
            }
        }
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[0][j] == 1) {
                cur[0] = 0;
            }
            for (int i = 1; i < m; i++) {
                if (obstacleGrid[i][j] != 1) {
                    cur[i] = cur[i - 1] + cur[i];
                } else {
                    cur[i] = 0;
                }
            }
        }
        return cur[m - 1];
    }

    public int uniquePaths(int m, int n) {
        int[] cur = new int[n];
        for (int i = 0; i < n; i++) {
            cur[i] = 1;
        }
        for (int j = 1; j < m; j++) {
            for (int i = 1; i < n; i++) {
                cur[i] = cur[i - 1] + cur[i];
            }
        }
        return cur[n - 1];
    }

    public int minPathSum2(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] cur = new int[m];
        cur[0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            cur[i] = cur[i - 1] + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            cur[0] += grid[0][j];
            for (int i = 1; i < m; i++) {
                cur[i] = Math.min(cur[i - 1], cur[i]) + grid[i][j];
            }
        }
        return cur[m - 1];
    }

    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] minNum = new int[m][n];
        minNum[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            minNum[i][0] = minNum[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            minNum[0][j] = minNum[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                minNum[i][j] = Math.min(minNum[i - 1][j], minNum[i][j - 1]) + grid[i][j];
            }
        }
        return minNum[m - 1][n - 1];
    }

    public ListNode deleteDuplicates3(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode result = new ListNode(0);
        result.next = head;
        ListNode q = result, p = head.next;
        boolean b = false;
        while (p != null) {
            if (q.next.val == p.val) {
                b = true;
            } else {
                if (b) {
                    b = false;
                    q.next = p;
                } else {
                    q = q.next;
                }
            }
            p = p.next;
        }
        if (b) {
            q.next = p;
        }
        return result.next;
    }

    public boolean isValidBST(TreeNode root) {
        return isValidBSTLoop(root, null, null);
    }

    private boolean isValidBSTLoop(TreeNode root, TreeNode minNode, TreeNode maxNode) {
        if (root == null) {
            return true;
        }
        if (minNode != null && root.val <= minNode.val || maxNode != null && root.val >= maxNode.val) {
            return false;
        }
        return isValidBSTLoop(root.left, minNode, root) && isValidBSTLoop(root.right, root, maxNode);
    }

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new LinkedList<>();
        }
        List<List<TreeNode>> lists = new LinkedList<>();
        List<TreeNode> list0 = new LinkedList<>();
        list0.add(null);
        lists.add(list0);
        List<TreeNode> list1 = new LinkedList<>();
        TreeNode t = new TreeNode(0);
        t.left = t.right = null;
        list1.add(t);
        lists.add(list1);

        for (int i = 2; i <= n; i++) {
            List<TreeNode> list = new LinkedList<>();
            for (int j = 0; j < i; j++) {
                List<TreeNode> l1 = lists.get(j);
                List<TreeNode> l2 = lists.get(i - j - 1);
                for (TreeNode t1 :
                        l1) {
                    for (TreeNode t2 :
                            l2) {
                        TreeNode root = new TreeNode(0);
                        root.left = t1;
                        root.right = t2;
                        list.add(root);
                    }
                }
            }
            lists.add(list);
        }

        List<TreeNode> result = new LinkedList<>();
        for (TreeNode item :
                lists.get(n)) {
            result.add(copyTreeNode(item));
        }
        for (TreeNode item :
                result) {
            value = 1;
            setValue(item);
        }

        return result;
    }

    private TreeNode copyTreeNode(TreeNode t) {
        if (t == null) {
            return null;
        }
        TreeNode root = new TreeNode(t.val);
        root.left = copyTreeNode(t.left);
        root.right = copyTreeNode(t.right);
        return root;
    }

    private int value = 1;

    private void setValue(TreeNode root) {
        if (root == null) {
            return;
        }
        setValue(root.left);
        root.val = value++;
        setValue(root.right);
    }

    public int numTrees(int n) {
        int[] nums = new int[n + 2];
        nums[0] = nums[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                nums[i] += nums[j] * nums[i - 1 - j];
            }
        }
        return nums[n];
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        inorderTraversalLoop(result, root);
        return result;
    }

    public void inorderTraversalLoop(List<Integer> result, TreeNode root) {
        if (root == null) {
            return;
        }
        inorderTraversalLoop(result, root.left);
        result.add(root.val);
        inorderTraversalLoop(result, root.right);
    }

    public List<String> restoreIpAddresses(String s) {
        List<String> result = new ArrayList<>();
        String[] str = new String[4];
        restoreIpAddressesLoop(result, s, str, 0, 0);
        return result;
    }

    public void restoreIpAddressesLoop(List<String> result, String s, String[] str, int num, int index) {
        if (num == str.length && index == s.length()) {
            StringBuilder sb = new StringBuilder(str[0]);
            for (int i = 1; i < str.length; i++) {
                sb.append(".").append(str[i]);
            }
            result.add(sb.toString());
            return;
        }
        if (!(num < str.length && index < s.length())) {
            return;
        }
        str[num] = s.substring(index, index + 1);
        restoreIpAddressesLoop(result, s, str, num + 1, index + 1);
        str[num] = "";

        if (index <= s.length() - 2) {
            int t = Integer.valueOf(s.substring(index, index + 2));
            if (t >= 10) {
                str[num] = s.substring(index, index + 2);
                restoreIpAddressesLoop(result, s, str, num + 1, index + 2);
                str[num] = "";
            }
        }

        if (index <= s.length() - 3) {
            int t = Integer.valueOf(s.substring(index, index + 3));
            if (t >= 100 && t <= 255) {
                str[num] = s.substring(index, index + 3);
                restoreIpAddressesLoop(result, s, str, num + 1, index + 3);
                str[num] = "";
            }
        }
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        int[] nums = new int[n - m + 1];
        ListNode p = head, q = p;
        int i = 1, j = nums.length - 1;
        while (i <= n) {
            if (i == m) {
                q = p;
            }
            if (i >= m) {
                nums[j--] = p.val;
            }
            p = p.next;
            i++;
        }
        for (int item :
                nums) {
            q.val = item;
            q = q.next;
        }
        return head;
    }

    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
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
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        subsetsWithDupLoop(result, new ArrayList<>(), nums, 0);
        return result;
    }

    private void subsetsWithDupLoop(List<List<Integer>> result, List<Integer> list, int[] nums, int from) {
        result.add(new ArrayList<>(list));
        for (int i = from; i < nums.length; i++) {
            if (i > from && nums[i] == nums[i - 1]) continue; // skip duplicates
            list.add(nums[i]);
            subsetsWithDupLoop(result, list, nums, i + 1);
            list.remove(list.size() - 1);
        }
    }

    public List<Integer> grayCode(int n) {
        List<Integer> result = new LinkedList<>();
        result.add(0);
        int i, j, k = 1;
        for (i = 0; i < n; i++) {
            for (j = k - 1; j >= 0; j--) {
                result.add(k + result.get(j));
            }
            k = 2 * k;
        }
        return result;
    }

    public ListNode partition(ListNode head, int x) {
        ListNode p = head, q, t;
        while (p != null && p.val < x) {
            p = p.next;
        }
        if (p == null) {
            return head;
        }
        q = p.next;
        while (q != null) {
            while (q != null && q.val >= x) {
                q = q.next;
            }
            if (q == null) {
                break;
            }
            t = p.next;
            int t1 = q.val, t2;
            while (p != q) {
                t2 = p.val;
                p.val = t1;
                t1 = t2;
                p = p.next;
            }
            p.val = t1;
            p = t;
            q = q.next;
        }
        return head;
    }

    public ListNode deleteDuplicates2(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        int pre = Integer.MIN_VALUE;
        while (head != null) {
            if (pre == Integer.MIN_VALUE || head.val != pre) {
                stack.push(head.val);
                pre = head.val;
            } else if (!stack.empty() && stack.peek() == pre) {
                stack.pop();
            }
            head = head.next;
        }
        ListNode p = new ListNode(0);
        p.next = null;
        while (!stack.empty()) {
            ListNode t = new ListNode(stack.pop());
            t.next = p.next;
            p.next = t;
        }
        return p.next;
    }

    public boolean search(int[] nums, int target) {
        int k = nums.length;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] > nums[i + 1]) {
                k = i + 1;
                break;
            }
        }
        int from, end, mid;
        if (nums[0] < target) {
            from = 0;
            end = k - 1;
        } else {
            from = k;
            end = nums.length - 1;
        }
        while (from <= end) {
            mid = (from + end) / 2;
            if (nums[mid] == target) {
                return true;
            } else if (nums[mid] > target) {
                end = mid - 1;
            } else {
                from = mid + 1;
            }
        }
        return false;
    }

    public int removeDuplicates2(int[] nums) {
        if (nums.length == 1) {
            return nums.length;
        }
        int i = 1, j = 1, tag = 1;
        while (j < nums.length) {
            if (nums[i - 1] == nums[j]) {
                if (tag == 1) {
                    nums[i] = nums[j];
                    tag = 2;
                    i++;
                    j++;
                } else {
                    j++;
                    tag = 1;
                    while (j < nums.length && nums[j - 1] == nums[j]) {
                        j++;
                    }
                }
            } else {
                nums[i] = nums[j];
                tag = 1;
                i++;
                j++;
            }
        }
        return i;
    }

    public boolean exist2(char[][] board, String word) {
        if (board.length == 0) return false;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                // 从i,j点作为起点开始搜索
                boolean isExisted = exist2Loop(board, i, j, word, 0);
                if (isExisted) return true;
            }
        }
        return false;
    }

    private boolean exist2Loop(char[][] board, int i, int j, String word, int idx) {
        if (idx >= word.length()) return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(idx))
            return false;
        // 将已经搜索过的字母标记一下，防止循环。只要变成另外一个字符，就不会再有循环了。
        board[i][j] ^= 255;
        boolean res = exist2Loop(board, i - 1, j, word, idx + 1) || exist2Loop(board, i + 1, j, word, idx + 1) || exist2Loop(board, i, j - 1, word, idx + 1) || exist2Loop(board, i, j + 1, word, idx + 1);
        // 再次异或255就能恢复成原来的字母
        board[i][j] ^= 255;
        return res;
    }

    public boolean exist(char[][] board, String word) {
        int m = board.length, n = board[0].length, k;
        int[] tag = new int[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] != word.charAt(0)) {
                    continue;
                }
                k = i * n + j;
                for (int t = 0; t < tag.length; t++) {
                    tag[t] = 0;
                }
                tag[k] = 1;
                boolean b = existLoop(board, tag, word, 0, k);
                if (b) {
                    return b;
                }
            }
        }
        return false;
    }

    public boolean existLoop(char[][] board, int[] tag, String word, int index, int from) {
        if (index == word.length() - 1) {
            return true;
        }
        int m = board.length, n = board[0].length;
        int[] dir = new int[4];
        dir[0] = from - n >= 0 ? from - n : from;   //上
        dir[1] = from + n < m * n ? from + n : from;   //下
        dir[2] = from - 1 >= (from / n) * n ? from - 1 : from;   //左
        dir[3] = from + 1 < (from / n + 1) * n ? from + 1 : from;   //右

        for (int to :
                dir) {
            int i = to / n, j = to - i * n;
            if (to == from || tag[to] == 1 || board[i][j] != word.charAt(index + 1)) {
                continue;
            }
            tag[to] = 1;
            boolean bool = existLoop(board, tag, word, index + 1, to);
            tag[to] = 0;
            if (bool) {
                return true;
            }
        }
        return false;
    }

    public List<List<Integer>> subsets2(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        subsetsLoop2(result, new LinkedList<>(), nums, 0);
        return result;
    }

    public void subsetsLoop2(List<List<Integer>> result, List<Integer> list, int[] nums, int index) {
        result.add(new LinkedList<>(list));
        for (int i = index; i < nums.length; i++) {
            list.add(nums[i]);
            subsetsLoop2(result, list, nums, i + 1);
            list.remove(list.size() - 1);
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        subsetsLoop(result, new ArrayList<>(), nums, 0);
        return result;
    }

    public void subsetsLoop(List<List<Integer>> result, List<Integer> list, int[] nums, int index) {
        if (index == nums.length) {
            result.add(new ArrayList<>(list));
            return;
        }
        subsetsLoop(result, list, nums, index + 1);
        list.add(nums[index]);
        subsetsLoop(result, list, nums, index + 1);
        list.remove(list.size() - 1);
    }


    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        combineLoop(result, new ArrayList<>(), n, k, 1);
        return result;
    }

    public void combineLoop(List<List<Integer>> result, List<Integer> list, int n, int k, int index) {
        if (list.size() == k) {
            result.add(new ArrayList<>(list));
            return;
        }
        for (int i = index; i <= n - k + list.size() + 1; i++) {
            list.add(i);
            combineLoop(result, list, n, k, i + 1);
            list.remove(list.size() - 1);
        }
    }

    public void sortColors2(int[] nums) {
        int start = 0, end = nums.length - 1, k = 0;
        while (k <= end) {
            if (nums[k] == 0) {
                swap(nums, start++, k++);
            } else if (nums[k] == 1) {
                k++;
            } else {
                swap(nums, k, end--);
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    public void sortColors(int[] nums) {
        int a = 0, b = 0, c = 0;
        for (int num :
                nums) {
            if (num == 0) {
                a++;
            } else if (num == 1) {
                b++;
            } else {
                c++;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (i < a) {
                nums[i] = 0;
            } else if (i < a + b) {
                nums[i] = 1;
            } else {
                nums[i] = 2;
            }
        }
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0) {
            return false;
        }
        int i, j, m = matrix.length, n = matrix[0].length;
        int from = 0, end = m * n - 1, mid;
        while (from <= end) {
            mid = (from + end) / 2;
            i = mid / n;
            j = mid - i * n;
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                end = mid - 1;
            } else {
                from = mid + 1;
            }
        }
        return false;
    }

    public void setZeroes2(int[][] matrix) {
        boolean[] flag = new boolean[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            boolean tag = false;
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 0) {
                    tag = true;
                    flag[j] = true;
                }
            }
            if (tag) {
                for (int j = 0; j < matrix[i].length; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int j = 0; j < flag.length; j++) {
            if (!flag[j]) {
                continue;
            }
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][j] = 0;
            }
        }
    }

    public void setZeroes(int[][] matrix) {
        Set<Integer> rows = new HashSet<>();
        Set<Integer> cols = new HashSet<>();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 0) {
                    rows.add(i);
                    cols.add(j);
                }
            }
        }
        for (int row :
                rows) {
            for (int j = 0; j < matrix[row].length; j++) {
                matrix[row][j] = 0;
            }
        }
        for (int col :
                cols) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][col] = 0;
            }
        }
    }

    public String simplifyPath(String path) {
        Stack<String> stack = new Stack<>();
        String[] paths = path.split("/");
        for (String item :
                paths) {
            if (item.equals("") || item.equals(".")) {
                continue;
            }
            if (item.equals("..")) {
                if (!stack.empty()) {
                    stack.pop();
                }
            } else {
                stack.push(item);
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.empty()) {
            sb.insert(0, stack.pop());
            sb.insert(0, "/");
        }
        return sb.length() == 0 ? "/" : sb.toString();
    }

    public int climbStairs2(int n) {
        int first = 1, second = 1, current = 1;
        for (int i = 2; i <= n; i++) {
            current = first + second;
            first = second;
            second = current;
        }
        return current;
    }

    public int climbStairs(int n) {
        int[] nums = new int[n + 1];
        nums[0] = nums[1] = 1;
        for (int i = 2; i <= n; i++) {
            nums[i] = nums[i - 1] + nums[i - 2];
        }
        return nums[n];
    }

    public String getPermutation(int n, int k) {
        int[] nums = new int[n];
        for (int i = 0; i < nums.length; i++) {
            nums[i] = i + 1;
        }
        while (k-- > 1) {
            nextPermutation(nums);
        }
        StringBuilder sb = new StringBuilder();
        for (int i :
                nums) {
            sb.append(i);
        }
        return sb.toString();
    }

    public int[][] generateMatrix(int n) {
        int[][] result = new int[n][n];
        int i, j, num = 1;
        for (i = 0; i < n; i++) {
            for (j = i; j < n - i - 1; j++)
                result[i][j] = num++;
            for (j = i; j < n - i - 1; j++)
                result[j][n - i - 1] = num++;
            for (j = i; j < n - i - 1; j++)
                result[n - i - 1][n - j - 1] = num++;
            for (j = i; j < n - i - 1; j++)
                result[n - j - 1][i] = num++;
        }
        if (n % 2 == 1)
            result[n / 2][n / 2] = num;
        return result;
    }

    public List<Interval> merge(List<Interval> intervals) {
        intervals.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start - o2.start;
            }
        });
        List<Interval> result = new ArrayList<>();
        int i;
        for (Interval interval :
                intervals) {
            for (i = 0; i < result.size(); i++) {
                if (result.get(i).end > interval.start) {
                    result.get(i).end = Math.max(result.get(i).end, interval.end);
                    break;
                }
            }
            if (i >= result.size()) {
                result.add(interval);
            }
        }
        return result;
    }

    public boolean canJump(int[] nums) {
        int i = nums.length - 2, k = nums.length - 1;
        while (i >= 0) {
            if (i + nums[i] >= k) {
                k = i;
            }
            i--;
        }
        return k == 0;
    }

    public double myPow(double x, int n) {
        if (n == 0) {
            return 1.0;
        }
        if (n == 1) {
            return x;
        }
        if (n < 0) {
            x = 1.0 / x;
            n = -n;
        }
        double result = 1.0;
        if (n % 2 == 1) {
            result *= x;
            n--;
        }
        result *= myPow(x * x, n / 2);
        return result;
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new LinkedList<>();
        Map<String, Integer> map = new HashMap<>();
        for (String str :
                strs) {
            char[] chs = str.toCharArray();
            Arrays.sort(chs);
            String s = String.valueOf(chs);
            if (!map.containsKey(s)) {
                map.put(s, result.size());
                List<String> list = new LinkedList<>();
                list.add(str);
                result.add(list);
            } else {
                List<String> list = result.get(map.get(s));
                list.add(str);
            }
        }
        return result;
    }

    public void rotate(int[][] matrix) {
        int i, j, t;
        for (i = 0; i < matrix.length / 2; i++) {
            for (j = i; j < matrix.length - i - 1; j++) {
                t = matrix[matrix.length - j - 1][i];
                matrix[matrix.length - j - 1][i] = matrix[matrix.length - i - 1][matrix.length - j - 1];
                matrix[matrix.length - i - 1][matrix.length - j - 1] = matrix[j][matrix.length - i - 1];
                matrix[j][matrix.length - i - 1] = matrix[i][j];
                matrix[i][j] = t;
            }
        }
    }

    //字典序
    public List<List<Integer>> permute2(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new LinkedList<>();
        Integer[] items = new Integer[nums.length];
        for (int i = 0; i < nums.length; i++) {
            items[i] = nums[i];
        }
        result.add(new LinkedList<>(Arrays.asList(items)));
        func(result, items);
        return result;
    }

    public void func(List<List<Integer>> result, Integer[] nums) {
        int t, l = Integer.MAX_VALUE, k = findK(nums);
        if (k == -1) {
            return;
        }
        for (int i = k + 1; i < nums.length; i++) {
            if (nums[i] > nums[k] && (l == Integer.MAX_VALUE || nums[i] <= nums[l])) {
                l = i;
            }
        }
        t = nums[k];
        nums[k] = nums[l];
        nums[l] = t;
        reverse(nums, k + 1);
        result.add(new LinkedList<>(Arrays.asList(nums)));
        func(result, nums);
    }

    public int findK(Integer[] nums) {
        int k = nums.length - 2;
        while (k >= 0 && nums[k] >= nums[k + 1]) {
            k--;
        }
        return k;
    }

    public void reverse(Integer[] nums, int from) {
        int i = from, j = nums.length - 1, t;
        while (i < j) {
            t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;
            i++;
            j--;
        }
    }

    public List<List<Integer>> permuteUnique2(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(nums);
        permuteUniqueLoop2(list, new ArrayList<>(), nums, new boolean[nums.length]);
        return list;
    }

    private void permuteUniqueLoop2(List<List<Integer>> list, List<Integer> tempList, int[] nums, boolean[] used) {
        if (tempList.size() == nums.length) {
            list.add(new ArrayList<>(tempList));
        } else {
            for (int i = 0; i < nums.length; i++) {
                if (used[i] || i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;
                used[i] = true;
                tempList.add(nums[i]);
                permuteUniqueLoop2(list, tempList, nums, used);
                used[i] = false;
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        Set<List<Integer>> result = new HashSet<>();
        permuteUniqueLoop(result, new LinkedList<>(), nums);
        return new LinkedList<>(result);
    }

    public void permuteUniqueLoop(Set<List<Integer>> result, List<Integer> list, int[] nums) {
        if (list.size() == nums.length) {
            List<Integer> l = new LinkedList<>();
            for (Integer i :
                    list) {
                l.add(nums[i]);
            }
            result.add(new LinkedList<>(l));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (list.contains(i)) {
                continue;
            }
            list.add(i);
            permuteUniqueLoop(result, list, nums);
            list.remove(list.size() - 1);
        }
    }

    public List<List<Integer>> permute3(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        permuteLoop3(list, new ArrayList<>(), nums);
        return list;
    }

    private void permuteLoop3(List<List<Integer>> list, List<Integer> tempList, int[] nums) {
        if (tempList.size() == nums.length) {
            list.add(new ArrayList<>(tempList));
        } else {
            for (int i = 0; i < nums.length; i++) {
                if (tempList.contains(nums[i])) continue; // element already exists, skip
                tempList.add(nums[i]);
                permuteLoop3(list, tempList, nums);
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    //递归
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Integer[] items = new Integer[nums.length];
        for (int i = 0; i < nums.length; i++) {
            items[i] = nums[i];
        }
        permuteLoop(result, items, 0);
        return result;
    }

    public void permuteLoop(List<List<Integer>> result, Integer[] nums, int index) {
        if (index == nums.length) {
            result.add(new ArrayList<>(Arrays.asList(nums)));
            return;
        }
        for (int i = index; i < nums.length; i++) {
            swap(nums, i, index);
            permuteLoop(result, nums, index + 1);
            swap(nums, i, index);
        }
    }

    public void swap(Integer[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    public String multiply2(String num1, String num2) {
        int[] num = new int[num1.length() + num2.length()];
        int a, b, c, t, k, tag = 0;
        for (int j = num2.length() - 1; j >= 0; j--) {
            b = num2.charAt(j) - '0';
            c = 0;
            k = num.length - (num2.length() - j);
            for (int i = num1.length() - 1; i >= 0; i--) {
                a = num1.charAt(i) - '0';
                t = a * b + c + num[k];
                num[k--] = t % 10;
                c = t / 10;
            }
            if (c != 0) {
                num[k] += c;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < num.length; i++) {
            if (tag == 0 && num[i] == 0) {
                continue;
            }
            tag = 1;
            sb.append(num[i]);
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> result = new LinkedList<>();
        combinationSum2Loop(result, new LinkedList<>(), candidates, target, 0, 0);
        return result;
    }

    public void combinationSum2Loop(List<List<Integer>> result, List<Integer> list, int[] candidates, int target, int from, int current) {
        if (current > target) {
            return;
        }
        if (current == target) {
            result.add(new LinkedList<>(list));
            return;
        }
        for (int i = from; i < candidates.length; i++) {
            if (i > from && candidates[i] == candidates[i - 1]) {
                continue;
            }
            list.add(candidates[i]);
            combinationSum2Loop(result, list, candidates, target, i + 1, current + candidates[i]);
            list.remove(list.size() - 1);
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> result = new LinkedList<>();
        combinationSumLoop(result, new LinkedList<>(), candidates, target, 0, 0);
        return result;
    }

    public void combinationSumLoop(List<List<Integer>> result, List<Integer> list, int[] candidates, int target, int from, int current) {
        if (current > target) {
            return;
        }
        if (current == target) {
            result.add(new LinkedList<>(list));
            return;
        }
        for (int i = from; i < candidates.length; i++) {
            list.add(candidates[i]);
            combinationSumLoop(result, list, candidates, target, i, current + candidates[i]);
            list.remove(list.size() - 1);
        }
    }

    public int[] searchRange(int[] nums, int target) {
        int[] result = {-1, -1};
        int from = 0, end = nums.length - 1, mid;
        while (from <= end) {
            mid = (from + end) / 2;
            int i = mid, j = mid;
            if (nums[mid] == target) {
                while (i > 0 && nums[i - 1] == nums[mid]) {
                    i--;
                }
                while (j < nums.length - 1 && nums[j + 1] == nums[mid]) {
                    j++;
                }
                result[0] = i;
                result[1] = j;
                break;
            } else if (nums[mid] > target) {
                end = i - 1;
            } else {
                from = j + 1;
            }
        }
        return result;
    }

    public int exist2Loop(int[] nums, int target) {
        int k = searchK(nums);
        int from, end, mid;
        if (k == -1) {
            from = 0;
            end = nums.length - 1;
        } else {
            if (nums[0] > target) {
                from = k;
                end = nums.length - 1;
            } else {
                from = 0;
                end = k - 1;
            }
        }
        while (from <= end) {
            mid = (from + end) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                from = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return -1;
    }

    public int searchK(int[] nums) {
        int from = 0, end = nums.length - 1, mid, k = -1;
        while (from < end) {
            mid = (from + end) / 2;
            if (mid > 0 && nums[mid] < nums[mid - 1]) {
                k = mid;
                break;
            }
            if (mid < nums.length - 1 && nums[mid] > nums[mid + 1]) {
                k = mid + 1;
                break;
            }
            if (nums[mid] > nums[from]) {
                from = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return k;
    }

    public void nextPermutation(int[] nums) {
        int t, l = Integer.MAX_VALUE, k = findK(nums);
        if (k == -1) {
            reverse(nums, 0);
            return;
        }
        for (int i = k + 1; i < nums.length; i++) {
            if (nums[i] > nums[k] && (l == Integer.MAX_VALUE || nums[i] <= nums[l])) {
                l = i;
            }
        }
        t = nums[k];
        nums[k] = nums[l];
        nums[l] = t;
        reverse(nums, k + 1);
    }

    public int findK(int[] nums) {
        int k = nums.length - 2;
        while (k >= 0 && nums[k] >= nums[k + 1]) {
            k--;
        }
        return k;
    }

    public void reverse(int[] nums, int from) {
        int i = from, j = nums.length - 1, t;
        while (i < j) {
            t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;
            i++;
            j--;
        }
    }

    public int divide(int dividend, int divisor) {
        return 0;
    }

    public ListNode swapPairs(ListNode head) {
        ListNode p = head, q = new ListNode(0), r = q;
        q.next = head;
        head = q;
        while (p != null && p.next != null) {
            q = p.next;
            p.next = q.next;
            q.next = p;
            r.next = q;
            r = p;
            p = p.next;
        }
        return head.next;
    }

    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        generateParenthesisLoop(result, "", 0, 0, n);
        return result;
    }

    public void generateParenthesisLoop(List<String> result, String s, int i, int j, int n) {
        if (s.length() == 2 * n) {
            result.add(s);
            return;
        }
        if (i < n) {
            generateParenthesisLoop(result, s + '(', i + 1, j, n);
        }
        if (j < i) {
            generateParenthesisLoop(result, s + ')', i, j + 1, n);
        }
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode p = head, q = new ListNode(0);   //删除q.next节点
        q.next = p;
        head = q;
        while (p != null) {
            if (n != 0) {
                n--;
            } else {
                q = q.next;
            }
            p = p.next;
        }
        q.next = q.next.next;
        return head.next;
    }

    public List<List<Integer>> fourSum3(int[] nums, int target) {
        Set<List<Integer>> result = new HashSet<>();
        Set<Integer> set = new HashSet<>();
        Arrays.sort(nums);
        for (int i = 1; i < nums.length - 2; i++) {
            set.add(nums[i - 1]);
            for (int j = i + 1; j < nums.length - 1; j++) {
                for (int k = j + 1; k < nums.length; k++) {
                    if (set.contains(target - nums[i] - nums[j] - nums[k])) {
                        Integer[] tt = {target - nums[i] - nums[j] - nums[k], nums[i], nums[j], nums[k]};
                        List<Integer> l = Arrays.asList(tt);
                        result.add(l);
                    }
                }
            }
        }
        return new ArrayList<>(result);
    }

    public List<List<Integer>> fourSum2(int[] nums, int target) {
        Set<List<Integer>> result = new HashSet<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; i++) {
            if (i != 0 && target != 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            for (int j = i + 1; j < nums.length - 2; j++) {
                List<List<Integer>> list = twoSum(nums, j + 1, target - nums[i] - nums[j]);
                for (List<Integer> l :
                        list) {
                    l.add(nums[i]);
                    l.add(nums[j]);
                    result.add(l);
                }
            }
        }
        return new ArrayList<>(result);
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        Set<List<Integer>> result = new HashSet<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; i++) {
            if (i != 0 && target != 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            for (int j = i + 1; j < nums.length - 2; j++) {
                int m = j + 1, n = nums.length - 1, t;
                while (m < n) {
                    t = nums[i] + nums[j] + nums[m] + nums[n];
                    if (t == target) {
                        Integer[] tt = {nums[i], nums[j], nums[m], nums[n]};
                        List<Integer> l = Arrays.asList(tt);
                        result.add(l);
                        m++;
                    } else if (t > target) {
                        n--;
                    } else {
                        m++;
                    }
                }
            }
        }
        return new ArrayList<>(result);
    }

    public List<String> letterCombinations(String digits) {
        List<String> result = new LinkedList<>();
        if (digits == null || digits.length() == 0) {
            return result;
        }
        String[] ss = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        letterCombinationsLoop(result, ss, digits, "", 0);
        return result;
    }

    public void letterCombinationsLoop(List<String> result, String[] ss, String digits, String str, int index) {
        if (index == digits.length()) {
            result.add(str);
            return;
        }
        String s = ss[digits.charAt(index) - '0'];
        for (int i = 0; i < s.length(); i++) {
            letterCombinationsLoop(result, ss, digits, str + s.charAt(i), index + 1);
        }
    }

    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int tag = 0, result = 0;
        for (int i = 0; i < nums.length - 2; i++) {
            if (i != 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int j = i + 1, k = nums.length - 1, t;
            while (j < k) {
                t = nums[i] + nums[j] + nums[k];
                if (tag == 0 || Math.abs(result - target) > Math.abs(t - target)) {
                    result = t;
                    tag = 1;
                }
                if (t > target) {
                    k--;
                } else {
                    j++;
                    while (j + 1 < nums.length && nums[j + 1] == nums[j])
                        j++;
                }
            }
        }
        return result;
    }

    public List<List<Integer>> threeSum3(int[] nums) {
        Set<List<Integer>> result = new HashSet<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i != 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            List<List<Integer>> l = twoSum(nums, i + 1, -nums[i]);
            for (List<Integer> item :
                    l) {
                item.add(nums[i]);
                result.add(item);
            }
        }
        return new ArrayList<>(result);
    }

    public List<List<Integer>> threeSum2(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i != 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            Set<Integer> set = new HashSet<>();
            for (int j = i + 1; j < nums.length; j++) {
                if (set.contains(-nums[i] - nums[j])) {
                    List<Integer> l = new ArrayList<>();
                    l.add(nums[i]);
                    l.add(-nums[i] - nums[j]);
                    l.add(nums[j]);
                    result.add(l);
                    while (j + 1 < nums.length && nums[j + 1] == nums[j]) {
                        j++;
                    }
                }
                set.add(nums[j]);
            }
        }
        return result;
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (!(p != null && q != null)) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int[] t = new int[m];
        int i = 0, j = 0, k = 0;
        System.arraycopy(nums1, 0, t, 0, m);
        while (i < m && j < n) {
            if (t[i] < nums2[j]) {
                nums1[k++] = t[i++];
            } else {
                nums1[k++] = nums2[j++];
            }
        }
        while (i < m) {
            nums1[k++] = t[i++];
        }
        while (j < n) {
            nums1[k++] = nums2[j++];
        }
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return null;
        }
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

    public String addBinary(String a, String b) {
        int l1 = a.length() - 1, l2 = b.length() - 1, c = 0, m, n;
        StringBuilder s = new StringBuilder();
        while (l1 >= 0 || l2 >= 0) {
            if (l1 >= 0 && l2 >= 0) {
                m = (a.charAt(l1) - '0') ^ (b.charAt(l2) - '0');
                n = (a.charAt(l1) - '0') & (b.charAt(l2) - '0');
            } else if (l2 < 0) {
                m = (a.charAt(l1) - '0');
                n = 0;
            } else {
                m = (b.charAt(l2) - '0');
                n = 0;
            }
            s.append(m ^ c);
            c = n ^ (m & c);
            l1--;
            l2--;
        }
        if (c != 0) {
            s.append(c);
        }
        return s.reverse().toString();
    }

    public int[] plusOne(int[] digits) {
        StringBuilder s = new StringBuilder();
        int c = 1, t;
        for (int i = digits.length - 1; i >= 0; i--) {
            t = digits[i] + c;
            s.append(t % 10);
            c = t / 10;
        }
        if (c != 0) {
            s.append(c);
        }
        int[] nums = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            nums[s.length() - i - 1] = s.charAt(i) - '0';
        }
        return nums;
    }

    public int lengthOfLastWord(String s) {
        int tag = 0, len = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) != ' ') {
                tag = 1;
                len++;
            } else if (tag == 1) {
                break;
            }
        }
        return len;
    }

    public int maxSubArray(int[] nums) {
        int sum, max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            sum = Math.max(nums[i], nums[i - 1] + nums[i]);
            max = Math.max(sum, max);
            nums[i] = sum;
        }
        return max;
    }

    public String countAndSay(int n) {
        StringBuffer sb = new StringBuffer("");
        String result = "1";
        int i, k = 0, l = n;
        System.out.println((l - n) + ":" + result);
        while (n-- > 1) {
            for (i = 0; i < result.length(); i++) {
                if (result.charAt(i) != result.charAt(k)) {
                    sb.append(i - k).append(result.charAt(k));
                    k = i;
                }
            }
            sb.append(i - k).append(result.charAt(k));
            result = sb.toString();
            sb = new StringBuffer("");
            k = 0;
            System.out.println((l - n + 1) + ":" + result);
        }
        return result;
    }

    public int searchInsert(int[] nums, int target) {
        int start = 0, end = nums.length - 1, middle = (start + end) / 2;
        while (start <= end) {
            middle = (start + end) / 2;
            if (nums[middle] < target) {
                start = middle + 1;
            } else if (nums[middle] > target) {
                end = middle - 1;
            } else {
                return middle;
            }
        }
        return nums[middle] > target ? middle : middle + 1;
    }

    public int strStr(String haystack, String needle) {
        return haystack.indexOf(needle);
    }

    public int removeElement(int[] nums, int val) {
        int i, j, k;
        for (i = 0, k = 0; i < nums.length; i++) {
            for (j = k; j < nums.length; j++) {
                if (val != nums[j]) {
                    nums[i] = nums[j];
                    k = j + 1;
                    break;
                }
            }
            if (j >= nums.length) {
                break;
            }
        }
        return i;
    }

    public int removeDuplicates(int[] nums) {
        int i, j, k;
        if (nums.length <= 1) {
            return nums.length;
        }
        for (i = 1, k = 0; i < nums.length; i++) {
            for (j = k + 1; j < nums.length; j++) {
                if (nums[i - 1] != nums[j]) {
                    nums[i] = nums[j];
                    k = j;
                    break;
                }
            }
            if (j >= nums.length) {
                break;
            }
        }
        return i;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode p = new ListNode(0), l = p;
        while (l1 != null && l2 != null) {
            l.next = new ListNode(0);
            l = l.next;
            if (l1.val < l2.val) {
                l.val = l1.val;
                l1 = l1.next;
            } else {
                l.val = l2.val;
                l2 = l2.next;
            }
        }
        while (l1 != null) {
            l.next = new ListNode(l1.val);
            l = l.next;
            l1 = l1.next;
        }
        while (l2 != null) {
            l.next = new ListNode(l2.val);
            l = l.next;
            l2 = l2.next;
        }
        l.next = null;
        return p.next;
    }

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            } else {
                if (stack.size() == 0) {
                    return false;
                }
                char ch = stack.peek();
                if (ch == '(' && c - ch == 1 || ch != '(' && c - ch == 2) {
                    stack.pop();
                } else {
                    return false;
                }
            }
        }
        return stack.size() == 0;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i != 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int j = i + 1, k = nums.length - 1, t;
            while (j < k) {
                t = nums[i] + nums[j] + nums[k];
                if (t > 0) {
                    k--;
                } else if (t < 0) {
                    j++;
                } else {
                    List<Integer> l = new ArrayList<>();
                    l.add(nums[i]);
                    l.add(nums[j]);
                    l.add(nums[k]);
                    result.add(l);
                    j++;
                    while (j < k && nums[j] == nums[j - 1]) {
                        j++;
                    }
                }
            }
        }
        return result;
    }

    public List<List<Integer>> twoSum(int[] nums, int from, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = from; i < nums.length; i++) {
            int temp = target - nums[i];
            if (map.containsKey(temp)) {
                List<Integer> list = new ArrayList<>();
                list.add(temp);
                list.add(nums[i]);
                result.add(list);
            }
            map.put(nums[i], i);
        }
        return result;
    }

    public String longestCommonPrefix(String[] strs) {
        int i, index = 0;
        //String str = "";
        for (i = 0; i < strs.length; i++) {
            if (index >= strs[i].length() || strs[i].charAt(index) != strs[0].charAt(index)) {
                break;
            }
            if (i == strs.length - 1) {
                index++;
                i = -1;
            }
        }
        return strs[0].substring(0, index);
    }

    public int maxArea(int[] height) {
        int s = 0, e = height.length - 1, max = 0;
        while (s < e) {
            max = Math.max(max, Math.min(height[s], height[e]) * (e - s));
            if (height[s] < height[e]) {
                s++;
            } else {
                e--;
            }
        }
        return max;
    }

    public boolean isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        }
        int y = 0;
        while (x > y) {
            y = y * 10 + x % 10;
            x /= 10;
        }
        return x == y || x == y / 10;
    }

    public int reverse(int x) {
        if (x == Integer.MIN_VALUE || x == 0) {
            return 0;
        }
        int tag = x > 0 ? 1 : -1;
        x = tag * x;
        String s = "";
        while (x > 0) {
            s = s + String.valueOf(x % 10);
            x = x / 10;
        }
        if (compareString(s)) {
            return 0;
        }
        return tag * Integer.parseInt(s);
    }

    private boolean compareString(String s) {
        String str = String.valueOf(Integer.MAX_VALUE);
        if (s.length() > str.length()) {
            return true;
        } else if (s.length() < str.length()) {
            return false;
        } else {
            return s.compareTo(str) > 0;
        }
    }

    public String longestPalindrome(String s) {
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenter(String s, int left, int right) {
        int l = left, r = right;
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
            l--;
            r++;
        }
        return r - l - 1;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        int i = 0, j = 0, k = 0, t = (m + n - 1) / 2;
        int[] nums = new int[nums1.length + nums2.length];
        while (i < m && j < n) {
            if (nums1[i] < nums2[j]) {
                nums[k] = nums1[i];
                i++;
            } else {
                nums[k] = nums2[j];
                j++;
            }
            k++;
        }
        while (i < m) {
            nums[k] = nums1[i];
            i++;
            k++;
        }
        while (j < n) {
            nums[k] = nums2[j];
            j++;
            k++;
        }
        if ((m + n) % 2 == 0) {
            return 1.0 * (nums[t] + nums[t + 1]) / 2;
        } else {
            return 1.0 * nums[t];
        }
    }

    public int lengthOfLongestSubstring(String s) {
        int i = 0, j, n = s.length(), len = 0;
        Map<Character, Integer> map = new HashMap<>();
        for (j = 0; j < n; j++) {
            if (map.containsKey(s.charAt(j))) {
                i = Math.max(map.get(s.charAt(j)), i);
            }
            len = Math.max(len, j - i + 1);
            map.put(s.charAt(j), j + 1);
        }
        return len;
    }

    public int lengthOfLongestSubstring2(String s) {
        int n = s.length(), ans = 0;
        int[] index = new int[128]; // current index of character
        // try to extend the range [i, j]
        for (int j = 0, i = 0; j < n; j++) {
            i = Math.max(index[s.charAt(j)], i);
            ans = Math.max(ans, j - i + 1);
            index[s.charAt(j)] = j + 1;
        }
        return ans;
    }

    public String multiply(String num1, String num2) {
        int[] num = new int[num1.length() + num2.length()];
        int a, b, c = 0, k, t;
        if (num1.length() < num2.length()) {
            String s = num1;
            num1 = num2;
            num2 = s;
        }
        for (int i = 0; i < num2.length(); i++) {
            k = num.length - i - 1;
            a = num2.charAt(num2.length() - i - 1) - '0';
            for (int j = 0; j < num1.length(); j++) {
                b = num1.charAt(num1.length() - j - 1) - '0';
                t = a * b + c + num[k];
                num[k] = t % 10;
                c = t / 10;
                k--;
            }
            if (c != 0) {
                num[k] = c;
                c = 0;
            }
        }
        int tag = 0;
        String result = "";
        for (int i = 0; i < num.length; i++) {
            if (tag == 0 && num[i] == 0) {
                continue;
            }
            if (tag == 0 && num[i] != 0) {
                tag = 1;
            }
            result = result + String.valueOf(num[i]);
        }
        if (tag == 0) {
            result = "0";
        }
        return result;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode l = new ListNode(0);
        l.next = null;
        ListNode q = l;
        int c = 0;
        while (l1 != null || l2 != null) {
            int x = (l1 != null) ? l1.val : 0;
            int y = (l2 != null) ? l2.val : 0;
            int t = x + y + c;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
            ListNode p = new ListNode(t % 10);
            p.next = null;
            c = t / 10;
            l.next = p;
            l = p;
        }
        if (c != 0) {
            ListNode p = new ListNode(c);
            p.next = null;
            l.next = p;
        }
        return q.next;
    }
}

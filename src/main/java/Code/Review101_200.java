package Code;

import Data.*;

import java.net.CookieHandler;
import java.net.HttpCookie;
import java.util.*;

public class Review101_200 {


    public int trailingZeroes(int n) {
        int res = 0;
        while (n > 0) {
            n /= 5;
            res += n;
        }
        return res;
    }

    public int titleToNumber(String s) {
        return s.equals("") ? 0 : s.charAt(s.length() - 1) - 'A' + 1 + 26 * titleToNumber(s.substring(0, s.length() - 1));
    }

    public int majorityElement2(int[] nums) {
        int major = nums[0], count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (count == 0) {
                count++;
                major = nums[i];
            } else if (major == nums[i]) {
                ++count;
            } else --count;
        }
        return major;
    }

    public int majorityElement(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            int n = map.getOrDefault(num, 0);
            if ((n + 1) > nums.length / 2) return num;
            map.put(num, n + 1);
        }
        return 0;
    }

    public String convertToTitle(int n) {
        return n == 0 ? "" : convertToTitle(--n / 26) + (char) ('A' + n % 26);
    }

    public int[] twoSum(int[] numbers, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(numbers[i])) {
                return new int[]{map.get(numbers[i]) + 1, i + 1};
            }
            map.put(target - numbers[i], i);
        }
        return new int[]{0, 0};
    }

    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        int i, j = 0;
        boolean tag = false;
        for (i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == ' ' && tag) {
                sb.insert(0, s.substring(j, i));
                sb.insert(0, " ");
                tag = false;
            }
            if (c != ' ' && !tag) {
                j = i;
                tag = true;
            }
        }
        if (tag) {
            sb.insert(0, s.substring(j, i));
            sb.insert(0, " ");
        }
        return sb.length() == 0 ? "" : sb.substring(1);
    }

    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode prev = head, slow = head, fast = head;
        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        prev.next = null;
        ListNode l1 = sortList(head);
        ListNode l2 = sortList(slow);
        return mergeList(l1, l2);
    }

    private ListNode mergeList(ListNode l1, ListNode l2) {
        ListNode l = new ListNode(0), p = l;
        while (l1 != null || l2 != null) {
            if (l2 == null || (l1 != null && l1.val < l2.val)) {
                p.next = l1;
                l1 = l1.next;
            } else {
                p.next = l2;
                l2 = l2.next;
            }
            p = p.next;
        }
        return l.next;
    }

    public List<Integer> postorderTraversal3(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        TreeNode temp;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        stack.push(root);
        while (!stack.empty()) {
            temp = stack.pop();
            if (!stack.empty() && temp == stack.peek()) {
                if (temp.right != null) {
                    stack.push(temp.right);
                    stack.push(temp.right);
                }
                if (temp.left != null) {
                    stack.push(temp.left);
                    stack.push(temp.left);
                }
            } else {
                res.add(temp.val);
            }
        }
        return res;
    }

    public List<Integer> postorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode prev = root;
        while (!stack.empty()) {
            TreeNode temp = stack.peek();
            if ((temp.left == null && temp.right == null) || (temp.right == null && prev == temp.left) || (prev == temp.right)) {
                res.add(temp.val);
                prev = temp;
                stack.pop();
            } else {
                if (temp.right != null) {
                    stack.push(temp.right);
                }
                if (temp.left != null) {
                    stack.push(temp.left);
                }
            }
        }
        return res;
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        postorderTraversalHelper(res, root);
        return res;
    }

    private void postorderTraversalHelper(List<Integer> res, TreeNode root) {
        if (root == null) return;
        postorderTraversalHelper(res, root.left);
        postorderTraversalHelper(res, root.right);
        res.add(root.val);
    }

    public List<Integer> preorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.empty()) {
            TreeNode temp = stack.pop();
            res.add(temp.val);
            if (temp.right != null) {
                stack.push(temp.right);
            }
            if (temp.left != null) {
                stack.push(temp.left);
            }
        }
        return res;
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        preorderTraversalHelper(res, root);
        return res;
    }

    private void preorderTraversalHelper(List<Integer> res, TreeNode root) {
        if (root == null) return;
        res.add(root.val);
        preorderTraversalHelper(res, root.left);
        preorderTraversalHelper(res, root.right);
    }

    public void reorderList(ListNode head) {
        if (head == null) return;
        ListNode prev = head;
        ListNode curr = head;
        while (curr.next != null && curr.next.next != null) {
            prev = prev.next;
            curr = curr.next.next;
        }
        ListNode part = new ListNode(0);
        part.next = prev.next;
        prev.next = null;
        curr = part.next;
        if (curr != null) {
            curr = curr.next;
            part.next.next = null;
            while (curr != null) {
                ListNode t = curr;
                curr = curr.next;
                t.next = part.next;
                part.next = t;
            }
        }
        prev = head;
        curr = part.next;
        while (prev != null && curr != null) {
            ListNode t = prev.next;
            prev.next = curr;
            curr = curr.next;
            prev.next.next = t;
            prev = t;
        }
    }

    public ListNode detectCycle(ListNode head) {
        if (head == null) return null;
        ListNode walker = head;
        ListNode runner = head;
        while (runner.next != null && runner.next.next != null) {
            walker = walker.next;
            runner = runner.next.next;
            if (walker == runner) break;
        }
        if (runner.next == null || runner.next.next == null) return null;
        walker = head;
        while (walker != runner) {
            walker = walker.next;
            runner = runner.next;
        }
        return walker;
    }

    public boolean hasCycle(ListNode head) {
        ListNode walker = new ListNode(0);
        ListNode runner = walker;
        walker.next = head;
        while (runner.next != null && runner.next.next != null) {
            walker = walker.next;
            runner = runner.next.next;
            if (walker == runner) return true;
        }
        return false;
    }

    public List<String> wordBreak2(String s, List<String> wordDict) {
        List<String> res = new ArrayList<>();
        wordBreakHelper(res, s, 0);
        return res;
    }

    private void wordBreakHelper(List<String> res, String s, int index) {

    }

    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean[] res = new boolean[n + 1];
        res[0] = true;
        for (int i = 1; i <= n; i++) {
            for (String word : wordDict) {
                if (word.length() <= i && res[i - word.length()] && s.substring(i - word.length(), i).equals(word)) {
                    res[i] = true;
                    break;
                }
            }
        }
        return res[n];
    }

    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) return null;

        Map<RandomListNode, RandomListNode> map = new HashMap<RandomListNode, RandomListNode>();

        // loop 1. copy all the nodes
        RandomListNode node = head;
        while (node != null) {
            map.put(node, new RandomListNode(node.label));
            node = node.next;
        }

        // loop 2. assign next and random pointers
        node = head;
        while (node != null) {
            map.get(node).next = map.get(node.next);
            map.get(node).random = map.get(node.random);
            node = node.next;
        }

        return map.get(head);
    }

    public int singleNumber2(int[] nums) {
        int one = 0, two = 0;
        for (int num : nums) {
            one = one ^ num & ~two;
            two = two ^ num & ~one;
        }
        return one;
    }

    public int singleNumber(int[] nums) {
        int res = 0;
        for (int num : nums) {
            res ^= num;
        }
        return res;
    }

    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) return 0;
        int n = ratings.length, res = 0;
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            nums[i] = 1;
        }
        for (int i = 0; i < n - 1; i++) {
            if (ratings[i] < ratings[i + 1]) {
                nums[i + 1] = nums[i] + 1;
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            if (i > 0 && ratings[i - 1] > ratings[i] && nums[i - 1] < nums[i] + 1) {
                nums[i - 1] = nums[i] + 1;
            }
            res += nums[i];
        }
        return res;
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length, res = 0, sum = 0, count = 0;
        for (int i = 0; i < n; i++) {
            sum += gas[i] - cost[i];
            if (sum < 0) {
                count += sum;
                sum = 0;
                res = i + 1;
            }
        }
        count += sum;
        return count < 0 ? -1 : res;
    }

    private Map<Integer, TempNode> tempNodeMap = new HashMap<>();

    public TempNode cloneGraph(TempNode node) {
        return clone(node);
    }

    private TempNode clone(TempNode node) {
        if (node == null) return null;
        if (tempNodeMap.containsKey(node.val)) {
            return tempNodeMap.get(node.val);
        }
        TempNode clone = new TempNode();
        clone.val = node.val;
        tempNodeMap.put(clone.val, clone);
        for (TempNode neighbor : node.neighbors) {
            TempNode tempNode = clone(neighbor);
            if (tempNode != null) {
                clone.neighbors.add(tempNode);
            }
        }
        return clone;
    }

    public int minCut2(String s) {
        int n = s.length();
        int[] dp = new int[n + 1];
        for (int i = 0; i <= n; i++) dp[i] = i - 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; i - j >= 0 && i + j < n && s.charAt(i - j) == s.charAt(i + j); j++) // odd length palindrome
                dp[i + j + 1] = Math.min(dp[i + j + 1], 1 + dp[i - j]);

            for (int j = 1; i - j + 1 >= 0 && i + j < n && s.charAt(i - j + 1) == s.charAt(i + j); j++) // even length palindrome
                dp[i + j + 1] = Math.min(dp[i + j + 1], 1 + dp[i - j + 1]);
        }
        return dp[n];
    }

    public int minCut(String s) {
        if (s.isEmpty()) return 0;
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int[] res = new int[n];
        for (int i = n - 1; i >= 0; i--) {
            res[i] = n - i - 1;
            for (int j = i; j < n; j++) {
                if (s.charAt(i) == s.charAt(j) && (j - i < 2 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                    if (j == n - 1)
                        res[i] = 0;
                    else if (res[j + 1] + 1 < res[i])
                        res[i] = res[j + 1] + 1;
                }
            }
        }
        return res[0];
    }

    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        partitionHelper(res, new ArrayList<>(), s, 0);
        return res;
    }

    private void partitionHelper(List<List<String>> res, List<String> list, String s, int index) {
        if (index >= s.length()) {
            res.add(new ArrayList<>(list));
        }
        for (int i = index; i < s.length(); i++) {
            boolean a = 2 * i <= s.length() + index - 2 && isSymmetry(s, index, 2 * i - index + 1);
            boolean b = 2 * i <= s.length() + index - 1 && isSymmetry(s, index, 2 * i - index);
            if (!(a || b)) continue;
            if (a) {
                list.add(s.substring(index, 2 * i - index + 2));
                partitionHelper(res, list, s, 2 * i - index + 2);
                list.remove(list.size() - 1);
            }
            if (b) {
                list.add(s.substring(index, 2 * i - index + 1));
                partitionHelper(res, list, s, 2 * i - index + 1);
                list.remove(list.size() - 1);
            }
        }
    }

    private boolean isSymmetry(String s, int lo, int hi) {
        int i = lo, j = hi;
        while (i < j) if (s.charAt(i++) != s.charAt(j--)) return false;
        return true;
    }

    public void solve(char[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0) return;
        int m = board.length, n = board[0].length, l = (Math.min(m, n) + 1) / 2;
        char[][] used = new char[m][n];
        for (int k = 0; k < l; k++) {
            for (int i = 0; i < m; i++) {

            }
        }
    }

    public int sumNumbers(TreeNode root) {
        if (root == null) return 0;
        return sumNumbersHelper(root, 0);
    }

    private int sumNumbersHelper(TreeNode root, int sum) {
        if (root.left == null && root.right == null) return 10 * sum + root.val;
        int res = 0;
        if (root.left != null) res += sumNumbersHelper(root.left, 10 * sum + root.val);
        if (root.right != null) res += sumNumbersHelper(root.right, 10 * sum + root.val);
        return res;
    }

    public int longestConsecutive(int[] nums) {
        //每次仅需要修改左右两边界值即可
        int res = 0, left, right, sum;
        Map<Integer, Integer> map = new HashMap<>();
        for (int item : nums) {
            if (map.containsKey(item)) continue;
            left = map.getOrDefault(item - 1, 0);
            right = map.getOrDefault(item + 1, 0);
            sum = left + right + 1;
            map.put(item, sum);
            map.put(item - left, sum);
            map.put(item + right, sum);
            res = Math.max(res, sum);
        }
        return res;
    }

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> lists = new ArrayList<>();
        int endIndex = -1;
        for (int i = 0; i < wordList.size(); i++) {
            if (endWord.equals(wordList.get(i))) {
                endIndex = i;
                break;
            }
        }
        if (endIndex == -1) return lists;

        Map<Integer, List<Integer>> map = new HashMap<>();
        List<Integer> beginList = new ArrayList<>();
        if (isTransferable(beginWord, wordList.get(0))) beginList.add(0);
        for (int i = 1; i < wordList.size(); i++) {
            if (isTransferable(beginWord, wordList.get(i))) beginList.add(i);
            List<Integer> list = new ArrayList<>();
            map.put(i, list);
            for (int j = 0; j < i; j++) {
                if (isTransferable(wordList.get(i), wordList.get(j))) {
                    list.add(j);
                }
            }
        }
        for (int begin : beginList) {
            List<String> list = new ArrayList<>();
            list.add(beginWord);
            list.add(endWord);
            findLaddersHelper(lists, list, wordList, map, begin, endIndex);
        }
        int len = Integer.MAX_VALUE;
        for (List<String> item : lists) {
            if (item.size() < len) len = item.size();
        }
        List<List<String>> res = new ArrayList<>();
        for (List<String> item : lists) {
            if (item.size() == len) res.add(item);
        }
        return res;
    }

    private void findLaddersHelper(List<List<String>> res, List<String> strs, List<String> wordList, Map<Integer, List<Integer>> map, int target, int index) {
        if (index == target) {
            res.add(new ArrayList<>(strs));
            System.out.println(strs);
            return;
        }
        List<Integer> list = map.get(index);
        if (list == null) return;
        for (int item : list) {
            strs.add(1, wordList.get(item));
            findLaddersHelper(res, strs, wordList, map, target, item);
            strs.remove(1);
        }
    }

    private boolean isTransferable(String s1, String s2) {
        int diff = 0;
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) != s2.charAt(i)) diff++;
        }
        return diff <= 1;
    }

    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            while (i < j && !isChar(s.charAt(i))) i++;
            while (i < j && !isChar(s.charAt(j))) j--;
            if (i < j && !isSame(s.charAt(i++), s.charAt(j--))) return false;
        }
        return true;
    }

    private boolean isChar(char c) {
        return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z';
    }

    private boolean isSame(char a, char b) {
        if (a >= '0' && a <= '9' || b >= '0' && b <= '9') return a == b;
        return a == b || Math.abs(a - b) == 32;
    }

    private int maxPathSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        maxPathSumHelper(root);
        return maxPathSum;
    }

    private int maxPathSumHelper(TreeNode root) {
        if (root == null) return 0;
        int left = Math.max(0, maxPathSumHelper(root.left));
        int right = Math.max(0, maxPathSumHelper(root.right));
        maxPathSum = Math.max(maxPathSum, left + right + root.val);
        return Math.max(left, right) + root.val;
    }

    public int maxProfit4(int[] prices) {
        if (prices.length <= 1) return 0;
        int K = 2;  //最大交易数目
        int max = 0;
        int[][] dp = new int[K + 1][prices.length];
        for (int k = 1; k <= K; k++) {
            int temp = dp[k - 1][0] - prices[0];
            for (int i = 1; i < prices.length; i++) {
                dp[k][i] = Math.max(dp[k][i - 1], prices[i] + temp);
                temp = Math.max(temp, dp[k - 1][i] - prices[i]);
                max = Math.max(dp[k][i], max);
            }
        }
        return max;
    }

    public int maxProfit3(int[] prices) {
        int hold1 = Integer.MIN_VALUE, hold2 = Integer.MIN_VALUE;
        int release1 = 0, release2 = 0;
        for (int i : prices) {                              // Assume we only have 0 money at first
            release2 = Math.max(release2, hold2 + i);     // The maximum if we've just sold 2nd stock so far.
            hold2 = Math.max(hold2, release1 - i);  // The maximum if we've just buy  2nd stock so far.
            release1 = Math.max(release1, hold1 + i);     // The maximum if we've just sold 1nd stock so far.
            hold1 = Math.max(hold1, -i);          // The maximum if we've just buy  1st stock so far.
        }
        return release2; ///Since release1 is initiated as 0, so release2 will always higher than release1.
    }

    public int maxProfit2(int[] prices) {
        int total = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i + 1] > prices[i]) total += prices[i + 1] - prices[i];
        }
        return total;
    }

    public int maxProfit(int[] prices) {
        int cur = Integer.MAX_VALUE, max = 0;
        for (int price : prices) {
            if (cur == Integer.MAX_VALUE) {
                cur = price;
                continue;
            }
            if (cur <= price) max = Math.max(max, price - cur);
            else cur = price;
        }
        return max;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] dp = new int[n + 1];
        dp[0] = triangle.get(0).get(0);
        dp[n] = dp[0];
        for (int i = 1; i < n; i++) {
            int min = Integer.MAX_VALUE;
            for (int j = i; j >= 0; j--) {
                dp[j] = triangle.get(i).get(j) +
                        Math.min(j == i ? Integer.MAX_VALUE : dp[j], j == 0 ? Integer.MAX_VALUE : dp[j - 1]);
                if (dp[j] < min) min = dp[j];
            }
            dp[n] = min;
        }
        return dp[n];
    }

    public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i <= rowIndex; i++) {
            res.add(1);
        }
        for (int i = 2; i < rowIndex; i++) {
            for (int j = i - 1; j >= 1; j--) {
                res.set(j, res.get(j) + res.get(j - 1));
            }
        }
        return res;
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        if (numRows == 0) return res;
        List<Integer> l = new ArrayList<>();
        l.add(1);
        res.add(l);
        for (int i = 1; i < numRows; i++) {
            List<Integer> list = new ArrayList<>();
            list.add(1);
            for (int j = 1; j < i; j++) {
                list.add(res.get(i - 1).get(j - 1) + res.get(i - 1).get(j));
            }
            list.add(1);
            res.add(list);
        }
        return res;
    }

    public void connect(TreeLinkNode root) {
        TreeLinkNode dummyHead = new TreeLinkNode(0);
        TreeLinkNode prev = dummyHead;
        while (root != null) {
            if (root.left != null) {
                prev.next = root.left;
                prev = prev.next;
            }
            if (root.right != null) {
                prev.next = root.right;
                prev = prev.next;
            }
            root = root.next;
            if (root == null) {
                prev = dummyHead;
                root = dummyHead.next;
                dummyHead.next = null;
            }
        }
    }

    public Node connect2(Node root) {
        Node curr = root;
        Node dummyHead = new Node();
        Node prev = dummyHead;
        while (curr != null) {
            if (curr.left != null) {
                prev.next = curr.left;
                prev = prev.next;
            }
            if (curr.right != null) {
                prev.next = curr.right;
                prev = prev.next;
            }
            curr = curr.next;
            if (curr == null) {
                prev = dummyHead;
                curr = dummyHead.next;
                dummyHead.next = null;
            }
        }
        return root;
    }

    public Node connect(Node root) {
        if (root == null) return null;
        Queue<Node> queue = new ArrayDeque<>();
        queue.add(root);
        Node prev = new Node();
        int count = 1, limit = 2;
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            if (count == limit) {
                limit *= 2;
                prev.next = null;
            } else {
                prev.next = node;
            }
            prev = node;
            if (node.left != null) queue.add(node.left);
            if (node.right != null) queue.add(node.right);
            count++;
        }
        prev.next = null;
        return root;
    }

    public int numDistinct(String s, String t) {
        int m = t.length(), n = s.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0) {
                    dp[i][j] = 1;
                    continue;
                }
                if (j == 0) {
                    dp[i][j] = 0;
                    continue;
                }
                dp[i][j] = dp[i][j - 1] + (t.charAt(i - 1) == s.charAt(j - 1) ? dp[i - 1][j - 1] : 0);
            }
        }
        return dp[m][n];
    }

    public void flatten2(TreeNode root) {
        TreeNode cur = root;
        while (cur != null) {
            if (cur.left != null) {
                TreeNode prev = cur.left;
                while (prev.right != null) {
                    prev = prev.right;
                }
                prev.right = cur.right;
                cur.right = cur.left;
                cur.left = null;
            }
            cur = cur.right;
        }
    }

    public void flatten(TreeNode root) {
        if (root == null) return;
        flatten(root.left);
        flatten(root.right);
        if (root.left != null) {
            TreeNode right = getRight(root.left);
            right.right = root.right;
            root.right = root.left;
            root.left = null;
        }
    }

    private TreeNode getRight(TreeNode root) {
        while (root.right != null) {
            root = root.right;
        }
        return root;
    }

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        pathSumHelper(res, new ArrayList<>(), root, sum);
        return res;
    }

    private void pathSumHelper(List<List<Integer>> res, List<Integer> list, TreeNode root, int sum) {
        if (root.left == null && root.right == null) {
            if (root.val == sum) {
                list.add(sum);
                res.add(new ArrayList<>(list));
                list.remove(list.size() - 1);
            }
            return;
        }
        list.add(root.val);
        if (root.left != null) pathSumHelper(res, list, root.left, sum - root.val);
        if (root.right != null) pathSumHelper(res, list, root.right, sum - root.val);
        list.remove(list.size() - 1);
    }

    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) return false;
        if (root.left == null && root.right == null && sum == root.val) return true;
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + (root.left == null && root.right == null ? 0 : Math.min(root.left == null ? Integer.MAX_VALUE : minDepth(root.left), root.right == null ? Integer.MAX_VALUE : minDepth(root.right)));
    }

    public boolean isBalanced(TreeNode root) {
        return getBSTHeight(root) != -1;
    }

    private int getBSTHeight(TreeNode root) {
        if (root == null) return 0;
        int leftHeight = getBSTHeight(root.left);
        if (leftHeight == -1) return -1;
        int rightHeight = getBSTHeight(root.right);
        if (rightHeight == -1) return -1;
        if (Math.abs(leftHeight - rightHeight) > 1) return -1;
        return Math.max(leftHeight, rightHeight) + 1;
    }

    private ListNode convertListToBSTHead;

    private int findSize(ListNode head) {
        ListNode ptr = head;
        int c = 0;
        while (ptr != null) {
            ptr = ptr.next;
            c += 1;
        }
        return c;
    }

    private TreeNode convertListToBST(int l, int r) {
        // Invalid case
        if (l > r) {
            return null;
        }

        int mid = (l + r) / 2;

        // First step of simulated inorder traversal. Recursively form
        // the left half
        TreeNode left = this.convertListToBST(l, mid - 1);

        // Once left half is traversed, process the current node
        TreeNode node = new TreeNode(this.convertListToBSTHead.val);
        node.left = left;

        // Maintain the invariance mentioned in the algorithm
        this.convertListToBSTHead = this.convertListToBSTHead.next;

        // Recurse on the right hand side and form BST out of them
        node.right = this.convertListToBST(mid + 1, r);
        return node;
    }

    public TreeNode sortedListToBST3(ListNode head) {
        // Get the size of the linked list first
        int size = this.findSize(head);

        this.convertListToBSTHead = head;

        // Form the BST now that we know the size
        return convertListToBST(0, size - 1);
    }

    public TreeNode sortedListToBST2(ListNode head) {
        if (head == null) return null;
        ListNode mid = findMiddleNode(head);
        TreeNode root = new TreeNode(mid.val);
        if (head == mid) return root;
        root.left = sortedListToBST2(head);
        root.right = sortedListToBST2(mid.next);
        return root;
    }

    private ListNode findMiddleNode(ListNode head) {
        ListNode prev = null, slow = head, fast = head;
        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        if (prev != null) prev.next = null;
        return slow;
    }

    public TreeNode sortedListToBST(ListNode head) {
        int n = 0;
        ListNode p = head;
        while (p != null) {
            n++;
            p = p.next;
        }
        int[] nums = new int[n];
        int i = 0;
        p = head;
        while (p != null) {
            nums[i++] = p.val;
            p = p.next;
        }
        return sortedArrayToBST(nums);
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTHelper(int[] nums, int lo, int hi) {
        if (lo > hi) return null;
        int mid = (lo + hi) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBSTHelper(nums, lo, mid - 1);
        root.right = sortedArrayToBSTHelper(nums, mid + 1, hi);
        return root;
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        levelOrderBottomHelper(res, root, 0);
        return res;
    }

    private void levelOrderBottomHelper(List<List<Integer>> res, TreeNode root, int depth) {
        if (root == null) return;
        if (res.size() <= depth) res.add(0, new ArrayList<>());
        levelOrderBottomHelper(res, root.left, depth + 1);
        levelOrderBottomHelper(res, root.right, depth + 1);
        res.get(res.size() - depth - 1).add(root.val);
    }

    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        if (inorder == null || inorder.length == 0) return null;
        return buildTree2Helper(inorder, postorder, 0, inorder.length, 0, postorder.length);
    }

    private TreeNode buildTree2Helper(int[] inorder, int[] postorder, int il, int ir, int pl, int pr) {
        if (il >= ir || pl >= pr) return null;
        TreeNode root = new TreeNode(postorder[pr - 1]);
        int index;
        for (index = il; index < ir; index++) if (inorder[index] == postorder[pr]) break;
        root.left = buildTree2Helper(inorder, postorder, il, index, pl, index - il + pl);
        root.right = buildTree2Helper(inorder, postorder, index + 1, ir, index - il + pl, pr - 1);
        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || preorder.length == 0) return null;
        return buildTreeHelper(preorder, inorder, 0, preorder.length, 0, inorder.length);
    }

    private TreeNode buildTreeHelper(int[] preorder, int[] inorder, int pl, int pr, int il, int ir) {
        if (pl >= pr || il >= ir) return null;
        TreeNode root = new TreeNode(preorder[pl]);
        int index;
        for (index = il; index < ir; index++) if (inorder[index] == preorder[pl]) break;
        root.left = buildTreeHelper(preorder, inorder, pl + 1, index - il + pl + 1, il, index);
        root.right = buildTreeHelper(preorder, inorder, index - il + pl + 1, pr, index + 1, ir);
        return root;
    }

    public int maxDepth(TreeNode root) {
        return root == null ? 0 : 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        zigzagLevelOrderHelper(res, root, 0);
        return res;
    }

    private void zigzagLevelOrderHelper(List<List<Integer>> res, TreeNode root, int depth) {
        if (root == null) return;
        if (res.size() <= depth) res.add(new ArrayList<>());
        List<Integer> list = res.get(depth);
        if (depth % 2 == 0) list.add(root.val);
        else list.add(0, root.val);
        zigzagLevelOrderHelper(res, root.left, depth + 1);
        zigzagLevelOrderHelper(res, root.right, depth + 1);
    }

    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        List<Integer> list = new ArrayList<>();
        List<TreeNode> temp = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        list.add(root.val);
        res.add(list);
        while (true) {
            temp = new ArrayList<>();
            list = new ArrayList<>();
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    temp.add(node.left);
                    list.add(node.left.val);
                }
                if (node.right != null) {
                    temp.add(node.right);
                    list.add(node.right.val);
                }
            }
            if (temp.size() == 0) break;
            res.add(list);
            queue.addAll(temp);
        }
        return res;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        levelOrderHelper(res, root, 0);
        return res;
    }

    private void levelOrderHelper(List<List<Integer>> res, TreeNode root, int depth) {
        if (root == null) return;
        if (res.size() <= depth) res.add(new ArrayList<>());
        res.get(depth).add(root.val);
        levelOrderHelper(res, root.left, depth + 1);
        levelOrderHelper(res, root.right, depth + 1);
    }

    public boolean isSymmetric(TreeNode root) {
        return root == null || isSymmetricHelper(root.left, root.right);
    }

    private boolean isSymmetricHelper(TreeNode left, TreeNode right) {
        if (left == null || right == null) return left == right;
        if (left.val != right.val) return false;
        return isSymmetricHelper(left.left, right.right) && isSymmetricHelper(left.right, right.left);
    }
}

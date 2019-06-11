package Code;

import java.util.*;

import Data.*;

/*
LeetCode 101-200
 */
public class LeetCode101_200 {


    public int numIslands(char[][] grid) {
        if (grid.length == 0) return 0;
        int m = grid.length, n = grid[0].length, num = 0;
        boolean[][] tags = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!tags[i][j] && grid[i][j] == '1') {
                    num++;
                    setChecked(grid, tags, i, j);
                }
                tags[i][j] = true;
            }
        }
        return num;
    }

    private void setChecked(char[][] grid, boolean[][] tags, int i, int j) {
        if (tags[i][j] || grid[i][j] == '0') {
            return;
        }
        tags[i][j] = true;
        if (i > 0) setChecked(grid, tags, i - 1, j);
        if (i < tags.length - 1) setChecked(grid, tags, i + 1, j);
        if (j > 0) setChecked(grid, tags, i, j - 1);
        if (j < tags[0].length - 1) setChecked(grid, tags, i, j + 1);
    }

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root != null) {
            rightSideLoop(list, root, 1);
        }
        return list;
    }

    private void rightSideLoop(List<Integer> list, TreeNode root, int height) {
        if (list.size() < height) {
            list.add(root.val);
        }
        if (root.right != null) {
            rightSideLoop(list, root.right, height + 1);
        }
        if (root.left != null) {
            rightSideLoop(list, root.left, height + 1);
        }
    }

    public int rob2(int[] nums) {
        if (nums.length == 0) return 0;
        int prev = 0, curr = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int temp = curr;
            curr = Math.max(prev + nums[i], curr);
            prev = temp;
        }
        return curr;
    }

    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        int[] dp = new int[nums.length + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i + 1] = Math.max(dp[i - 1] + nums[i], dp[i]);
        }
        return dp[nums.length];
    }

    public int hammingWeight(int n) {
        int num = 0;
        for (int i = 0; i < 32; i++) {
            num += n & 1;
            n >>>= 1;
        }
        return num;
    }

    public int reverseBits(int n) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            result += n & 1;
            n >>>= 1;   // CATCH: must do unsigned shift
            if (i < 31) // CATCH: for last digit, don't shift!
                result <<= 1;
        }
        return result;
    }

    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    private void reverse(int[] nums, int start, int end) {
        int i = start, j = end, t = 0;
        while (i < j) {
            t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;
            i++;
            j--;
        }
    }

    public List<String> findRepeatedDnaSequences2(String s) {
        Map<String, Boolean> map = new HashMap<>();
        List<String> result = new ArrayList<>();
        String str = "";
        for (int i = 0; i < s.length() - 9; i++) {
            str = s.substring(i, i + 10);
            if (map.containsKey(str)) {
                if (map.get(str)) {
                    result.add(str);
                } else {
                    map.replace(str, true);
                }
            } else {
                map.put(str, false);
            }
        }
        return result;
    }

    public List<String> findRepeatedDnaSequences(String s) {
        Set<String> set = new HashSet<>();
        Set<String> result = new HashSet<>();
        String str = "";
        for (int i = 0; i < s.length() - 9; i++) {
            str = s.substring(i, i + 10);
            if (set.contains(str)) {
                result.add(str);
            } else {
                set.add(str);
            }
        }
        return new ArrayList<>(result);
    }

    public String largestNumber(int[] nums) {
        Integer[] numbers = new Integer[nums.length];
        for (int i = 0; i < nums.length; i++) {
            numbers[i] = nums[i];
        }
        Comparator<Integer> comparator = new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                String s1 = String.valueOf(o1), s2 = String.valueOf(o2);
                int i = 0, j = 0;
                while (i < s1.length() && j < s2.length()) {
                    if (s1.charAt(i) != s2.charAt(j)) {
                        return s1.charAt(i) - s2.charAt(j);
                    } else {
                        if (i == s1.length() - 1 && j == s2.length() - 1) {
                            break;
                        }
                        if (i == s1.length() - 1) {
                            i = 0;
                        } else {
                            i++;
                        }
                        if (j == s2.length() - 1) {
                            j = 0;
                        } else {
                            j++;
                        }
                    }
                }
                return 0;
            }
        };
        Arrays.sort(numbers, comparator);
        StringBuilder str = new StringBuilder();
        int tag = 0;
        for (int i = numbers.length - 1; i >= 0; i--) {
            if (numbers[i] != 0 || tag != 0) {
                str.append(numbers[i]);
                tag = 1;
            }
        }
        return str.length() == 0 ? "0" : str.toString();
    }

    public int trailingZeroes2(int n) {
        int result = 0;
        while (n > 0) {
            n /= 5;
            result += n;
        }
        return result;
    }

    public int trailingZeroes1(int n) {
        if (n < 5) return 0;
        if (n < 10) return 1;
        return (n / 5 + trailingZeroes1(n / 5));
    }

    public int trailingZeroes(int n) {
        if (n < 5) return 0;
        int m = (int) Math.pow(5, (int) (Math.log(n) / Math.log(5)));
        return (m - 1) / 4 + trailingZeroes(n - m);
    }

    public int titleToNumber1(String s) {
        return s.equals("") ? 0 : s.charAt(s.length() - 1) - 'A' + 1 + 26 * titleToNumber1(s.substring(0, s.length() - 1));
    }

    public int titleToNumber(String s) {
        int n, result = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            n = s.charAt(i) - 'A' + 1;
            result += n * Math.pow(26, s.length() - 1 - i);
        }
        return result;
    }

    public int majorityElement(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                map.replace(num, map.get(num) + 1);
                if (map.get(num) > nums.length / 2) {
                    return num;
                }
            } else {
                map.put(num, 1);
            }
        }
        return 0;
    }

    public String convertToTitle1(int n) {
        return n == 0 ? "" : convertToTitle(--n / 26) + (char) ('A' + (n % 26));
    }

    public String convertToTitle(int k) {
        StringBuilder result = new StringBuilder();
        int m, n = k - 1;
        while (n >= 0) {
            m = n % 26;
            n = (n - m) / 26 - 1;
            result.insert(0, (char) (m + 'A'));
        }
        return result.toString();
    }

    public int[] twoSum(int[] numbers, int target) {
        int i = 0, j = numbers.length - 1, t;
        while (i < j) {
            t = numbers[i] + numbers[j];
            if (t == target) {
                break;
            } else if (t > target) {
                j--;
            } else {
                i++;
            }
        }
        return new int[]{i + 1, j + 1};
    }

    public String fractionToDecimal1(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        // "+" or "-"
        res.append(((numerator > 0) ^ (denominator > 0)) ? "-" : "");
        long num = Math.abs((long) numerator);
        long den = Math.abs((long) denominator);

        // integral part
        res.append(num / den);
        num %= den;
        if (num == 0) {
            return res.toString();
        }

        // fractional part
        res.append(".");
        HashMap<Long, Integer> map = new HashMap<Long, Integer>();
        map.put(num, res.length());
        while (num != 0) {
            num *= 10;
            res.append(num / den);
            num %= den;
            if (map.containsKey(num)) {
                int index = map.get(num);
                res.insert(index, "(");
                res.append(")");
                break;
            } else {
                map.put(num, res.length());
            }
        }
        return res.toString();
    }

    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        Map<Long, Integer> remainder = new HashMap<>();
        StringBuilder result = new StringBuilder();
        if ((numerator ^ denominator) < 0) {
            result.append("-");
        }
        long numerator1 = Math.abs((long) numerator);
        long denominator1 = Math.abs((long) denominator);
        if (numerator1 < denominator1) {
            result.append("0");
        }
        while (numerator1 >= denominator1) {
            result.append(numerator1 / denominator1);
            numerator1 %= denominator1;
        }
        if (numerator1 != 0) {
            result.append(".");
        }
        while (numerator1 != 0) {
            numerator1 *= 10;
            result.append(numerator1 / denominator1);
            if (remainder.containsKey(numerator1)) {
                result.replace(result.length() - 1, result.length(), ")");
                result.insert(remainder.get(numerator1), "(");
                break;
            }
            remainder.put(numerator1, result.length() - 1);
            numerator1 %= denominator1;
        }
        return result.toString();
    }

    public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\."), v2 = version2.split("\\.");
        int n1 = v1.length, n2 = v2.length;
        for (int j = n1 - 1; j >= 0; j--) {
            if (Integer.valueOf(v1[j]) == 0) {
                n1--;
            } else {
                break;
            }
        }
        for (int j = n2 - 1; j >= 0; j--) {
            if (Integer.valueOf(v2[j]) == 0) {
                n2--;
            } else {
                break;
            }
        }
        int i = 0, a, b;
        while (i < n1 && i < n2) {
            a = Integer.valueOf(v1[i]);
            b = Integer.valueOf(v2[i]);
            if (a == b) {
                i++;
            } else {
                return a - b > 0 ? 1 : -1;
            }
        }
        if (i >= n1 && i >= n2) {
            return 0;
        } else if (i < n1) {
            return 1;
        } else {
            return -1;
        }
    }

    public int findPeakElement(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            if (((i == 0 || nums[i] > nums[i - 1]) && i < nums.length - 1 && nums[i] > nums[i + 1])
                    || ((i == nums.length - 1 || nums[i] > nums[i + 1]) && i > 0 && nums[i] > nums[i - 1])) {
                return i;
            }
        }
        return 0;
    }

    /**
     * 两个单链表的起点
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p = headA, q = headB;
        if (p == null || q == null) {
            return null;
        }
        while (p != null || q != null) {
            if (p == null) {
                p = headB;
            }
            if (q == null) {
                q = headA;
            }
            if (p.val == q.val) {
                return p;
            }
            p = p.next;
            q = q.next;
        }
        return null;
    }

    public int findMin2(int[] nums) {
        int i = 0, j = nums.length - 1, k;
        while (i < j) {
            k = (i + j) / 2;
            if (nums[k] > nums[j]) {
                i = k + 1;
            } else if (nums[k] < nums[j]) {
                j = k;
            } else {
                j--;
            }
        }
        return nums[i];
    }

    public int findMin(int[] nums) {
        int i = 0, j = nums.length - 1, k;
        while (i < j) {
            k = (i + j) / 2;
            if (nums[i] > nums[j]) {
                if (nums[k] >= nums[i]) {
                    i = k + 1;
                } else {
                    j = k;
                }
            } else {
                break;
            }
        }
        return nums[i];
    }

    public int maxProduct(int[] nums) {
        // store the result that is the max we have found so far
        int r = nums[0];

        // imax/imin stores the max/min product of
        // subarray that ends with the current number A[i]
        for (int i = 1, imax = r, imin = r; i < nums.length; i++) {
            // multiplied by a negative makes big number smaller, small number bigger
            // so we redefine the extremums by swapping them
            if (nums[i] < 0) {
                int t = imax;
                imax = imin;
                imin = t;
            }
            // max/min product for the current number is either the current number itself
            // or the max/min by the previous number times the current one
            imax = Math.max(nums[i], imax * nums[i]);
            imin = Math.min(nums[i], imin * nums[i]);

            // the newly computed max value is a candidate for our global result
            r = Math.max(r, imax);
        }
        return r;
    }

    public String reverseWords(String s) {
        s = s.trim();
        StringBuilder sb = new StringBuilder();
        int i = 0, j = 0, k = 1;
        for (i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ' && k == 1) {
                sb.insert(0, s.substring(j, i));
                sb.insert(0, " ");
                k = 0;
            } else if (s.charAt(i) != ' ' && k == 0) {
                j = i;
                k = 1;
            }
        }
        sb.insert(0, s.substring(j, i));
        return sb.toString();
    }

    public int evalRPN(String[] tokens) {
        Stack<String> stack = new Stack<>();
        for (String item : tokens) {
            if ("+".equals(item)) {
                String str2 = stack.pop();
                String str1 = stack.pop();
                stack.push(String.valueOf(Integer.valueOf(str1) + Integer.valueOf(str2)));
            } else if ("-".equals(item)) {
                String str2 = stack.pop();
                String str1 = stack.pop();
                stack.push(String.valueOf(Integer.valueOf(str1) - Integer.valueOf(str2)));
            } else if ("*".equals(item)) {
                String str2 = stack.pop();
                String str1 = stack.pop();
                stack.push(String.valueOf(Integer.valueOf(str1) * Integer.valueOf(str2)));
            } else if ("/".equals(item)) {
                String str2 = stack.pop();
                String str1 = stack.pop();
                stack.push(String.valueOf(Integer.valueOf(str1) / Integer.valueOf(str2)));
            } else {
                stack.push(item);
            }
        }
        return Integer.valueOf(stack.pop());
    }

    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        //step 1. Cut the list to two halves.
        ListNode prev = null, slow = head, fast = head;
        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        prev.next = null;
        //step 2. Sort each half.
        ListNode l1 = sortList(head);
        ListNode l2 = sortList(slow);
        //step 3. Merge l1 and l2.
        return mergeListNode(l1, l2);
    }

    private ListNode mergeListNode(ListNode l1, ListNode l2) {
        ListNode l = new ListNode(0), p = l;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                p.next = l1;
                l1 = l1.next;
            } else {
                p.next = l2;
                l2 = l2.next;
            }
            p = p.next;
        }
        if (l1 != null) {
            p.next = l1;
        }
        if (l2 != null) {
            p.next = l2;
        }
        return l.next;
    }

    public ListNode insertionSortList(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode p = root.next, q, t;
        while (p.next != null) {
            if (p.next.val < p.val) {
                t = p.next;
                p.next = t.next;
                q = root;
                while (q != p) {
                    if (q.next.val > t.val) {
                        t.next = q.next;
                        q.next = t;
                        break;
                    } else {
                        q = q.next;
                    }
                }
            } else {
                p = p.next;
            }
        }
        return root.next;
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode t;
        if (root != null) {
            stack.push(root);
        }
        while (!stack.isEmpty()) {
            t = stack.pop();
            res.add(0, t.val);
            if (t.left != null) {
                stack.push(t.left);
            }
            if (t.right != null) {
                stack.push(t.right);
            }
        }
        return res;
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode t;
        if (root != null) {
            stack.push(root);
        }
        while (!stack.isEmpty()) {
            t = stack.pop();
            res.add(t.val);
            if (t.right != null) {
                stack.push(t.right);
            }
            if (t.left != null) {
                stack.push(t.left);
            }
        }
        return res;
    }

    /*
    Also, you can follow this.
    First, cut the linkedlist in half,
    Then, reverse the second half,
    Final, connect one by one from both halves.

    public void reorderList(ListNode head) {
        if (head == null || head.next == null) return;

        //Find the middle of the list
        ListNode p1 = head;
        ListNode p2 = head;
        while (p2.next != null && p2.next.next != null) {
            p1 = p1.next;
            p2 = p2.next.next;
        }

        //Reverse the half after middle  1->2->3->4->5->6 to 1->2->3->6->5->4
        ListNode preMiddle = p1;
        ListNode preCurrent = p1.next;
        while (preCurrent.next != null) {
            ListNode current = preCurrent.next;
            preCurrent.next = current.next;
            current.next = preMiddle.next;
            preMiddle.next = current;
        }

        //Start reorder one by one  1->2->3->6->5->4 to 1->6->2->5->3->4
        p1 = head;
        p2 = preMiddle.next;
        while (p1 != preMiddle) {
            preMiddle.next = p2.next;
            p2.next = p1.next;
            p1.next = p2;
            p1 = p2.next;
            p2 = preMiddle.next;
        }
    }
     */
    public void reorderList(ListNode head) {
        reorderPre = head;
        reorderListLoop(head, 0);
    }

    private int reorderNumLimit = 0;

    private ListNode reorderPre = null;

    private void reorderListLoop(ListNode node, int index) {
        if (node == null) {
            reorderNumLimit = index / 2;
            return;
        }
        reorderListLoop(node.next, index + 1);
        if (index == reorderNumLimit) {
            node.next = null;
            return;
        }
        if (index > reorderNumLimit) {
            node.next = reorderPre.next;
            reorderPre.next = node;
            reorderPre = node.next;
        }
    }

    public ListNode detectCycle(ListNode head) {
        if (head == null) return null;
        ListNode walker = head, runner = head;
        while (runner.next != null && runner.next.next != null) {
            walker = walker.next;
            runner = runner.next.next;
            if (walker == runner) break;
        }
        if (runner.next == null || runner.next.next == null) return null;
        runner = head;
        while (walker != runner) {
            walker = walker.next;
            runner = runner.next;
        }
        return walker;
    }

    /*
    Walker goes 1 step at a time,and runner goes 2 steps at a time.
    If we think walker is still,then runner goes 1 step at a time.
    So,the problem is just like a Chasing problem.
    There is a time when runner catches walker.
     */
    public boolean hasCycle2(ListNode head) {
        if (head == null) return false;
        ListNode walker = head;
        ListNode runner = head;
        while (runner.next != null && runner.next.next != null) {
            walker = walker.next;
            runner = runner.next.next;
            if (walker == runner) return true;
        }
        return false;
    }

    public boolean hasCycle(ListNode head) {
        while (head != null) {
            if (head.val != Integer.MIN_VALUE) {
                head.val = Integer.MIN_VALUE;
            } else {
                return true;
            }
            head = head.next;
        }
        return false;
    }

    public boolean wordBreak2(String s, List<String> wordDict) {
        boolean[] f = new boolean[s.length() + 1];
        f[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (String str : wordDict) {
                if (str.length() <= i && f[i - str.length()] && str.equals(s.substring(i - str.length(), i))) {
                    f[i] = true;
                    break;
                }
            }
        }
        return f[s.length()];
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        if (s.equals("")) {
            return true;
        }
        for (String str : wordDict) {
            if (s.length() >= str.length() && str.equals(s.substring(0, str.length()))) {
                boolean b = wordBreak(s.substring(str.length()), wordDict);
                if (b) {
                    return true;
                }
            }
        }
        return false;
    }

    private HashMap<Integer, RandomListNode> map2 = new HashMap<>();

    public RandomListNode copyRandomList(RandomListNode head) {
        RandomListNode node = copy(head);
        if (node != null) {
            node.random = copy(head.random);
            node.next = copyRandomList(head.next);
        }
        return node;
    }

    private RandomListNode copy(RandomListNode head) {
        if (head == null) return null;
        RandomListNode node;
        if (map2.containsKey(head.label)) {
            node = map2.get(head.label);
        } else {
            node = new RandomListNode(head.label);
            map2.put(head.label, node);
        }
        return node;
    }

    public int singleNumber4(int[] nums) {
        int ones = 0, twos = 0;
        for (int item : nums) {
            //出现三次的数：第一次只出现在ones中，第二次只出现在twos中，第三次都不出现。
            ones = (ones ^ item) & ~twos;
            twos = (twos ^ item) & ~ones;
        }
        return ones;
    }

    public int singleNumber3(int[] nums) {
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 1; i += 3) {
            if (nums[i] != nums[i + 1]) {
                return nums[i];
            }
        }
        return nums[nums.length - 1];
    }

    public int singleNumber2(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            nums[0] ^= nums[i];
        }
        return nums[0];
    }

    public int singleNumber(int[] nums) {
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 1; i += 2) {
            if (nums[i] != nums[i + 1]) {
                return nums[i];
            }
        }
        return nums[nums.length - 1];
    }

    public int canCompleteCircuit2(int[] gas, int[] cost) {
        int res = 0, sum = 0, total = 0, length = gas.length;
        for (int i = 0; i < length; i++) {
            sum += gas[i] - cost[i];
            if (sum < 0) {
                total += sum;
                sum = 0;
                res = i + 1;
            }
        }
        total += sum;
        return total < 0 ? -1 : res;
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        for (int i = 0; i < gas.length; i++) {
            int sum = 0;
            for (int j = i; j < gas.length; j++) {
                sum += gas[j] - cost[j];
                if (sum < 0) {
                    break;
                }
            }
            if (sum < 0) {
                continue;
            }
            for (int j = 0; j < i; j++) {
                sum += gas[j] - cost[j];
                if (sum < 0) {
                    break;
                }
            }
            if (sum >= 0) {
                return i;
            }
        }
        return -1;
    }

    private HashMap<Integer, UndirectedGraphNode> map = new HashMap<>();

    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        return clone(node);
    }

    private UndirectedGraphNode clone(UndirectedGraphNode node) {
        if (node == null) return null;
        if (map.containsKey(node.label)) {
            return map.get(node.label);
        }
        UndirectedGraphNode clone = new UndirectedGraphNode(node.label);
        map.put(clone.label, clone);
        for (UndirectedGraphNode neighbor : node.neighbors) {
            clone.neighbors.add(clone(neighbor));
        }
        return clone;
    }

    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        boolean[][] isPartition = new boolean[s.length()][s.length()];
        getPartition(s, isPartition);
        partitionLoop(res, new ArrayList<>(), isPartition, s, 0);
        return res;
    }

    private void partitionLoop(List<List<String>> res, List<String> path, boolean[][] isPartition, String s, int pos) {
        if (pos == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = pos; i < s.length(); i++) {
            if (isPartition[pos][i]) {
                path.add(s.substring(pos, i + 1));
                partitionLoop(res, path, isPartition, s, i + 1);
                path.remove(path.size() - 1);
            }
        }
    }

    private void getPartition(String s, boolean[][] isPartition) {
        for (int m = 0; m < s.length(); m++) {
            isPartition[m][m] = true;
            int i = m - 1, j = m + 1;
            while (i >= 0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
                isPartition[i][j] = true;
                i--;
                j++;
            }
            i = m;
            j = m + 1;
            while (i >= 0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
                isPartition[i][j] = true;
                i--;
                j++;
            }
        }
    }

    public void solve(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        char[][] result = new char[board.length][board[0].length];
        for (int j = 0; j < board[0].length; j++) {
            if (board[0][j] == 'O') {
                solveLoop(board, result, 0, j);
            }
            if (board[board.length - 1][j] == 'O') {
                solveLoop(board, result, board.length - 1, j);
            }
        }
        for (int i = 0; i < board.length; i++) {
            if (board[i][0] == 'O') {
                solveLoop(board, result, i, 0);
            }
            if (board[i][board[0].length - 1] == 'O') {
                solveLoop(board, result, i, board[0].length - 1);
            }
        }
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                board[i][j] = result[i][j] == 'O' ? 'O' : 'X';
            }
        }
    }

    private void solveLoop(char[][] board, char[][] result, int i, int j) {
        if (result[i][j] == 'O' || board[i][j] == 'X') {
            return;
        }
        result[i][j] = 'O';
        //上
        if (i > 0) {
            solveLoop(board, result, i - 1, j);
        }
        //下
        if (i < board.length - 1) {
            solveLoop(board, result, i + 1, j);
        }
        //左
        if (j > 0) {
            solveLoop(board, result, i, j - 1);
        }
        //右
        if (j < board[0].length - 1) {
            solveLoop(board, result, i, j + 1);
        }
    }

    public int sumNumbers(TreeNode root) {
        if (root != null) {
            sumNumbersLoop(root, root.val);
        }
        return result;
    }

    private int result = 0;

    private void sumNumbersLoop(TreeNode root, int sum) {
        if (root.left == null && root.right == null) {
            result += sum;
            return;
        }
        if (root.left != null) {
            sumNumbersLoop(root.left, sum * 10 + root.left.val);
        }
        if (root.right != null) {
            sumNumbersLoop(root.right, sum * 10 + root.right.val);
        }
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        return 0;
    }

    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            char a = s.charAt(i), b = s.charAt(j);
            while (i < j && !isChar(a) && !isNum(a)) {
                i++;
                a = s.charAt(i);
            }
            while (i < j && !isChar(b) && !isNum(b)) {
                j--;
                b = s.charAt(j);
            }
            if (i < j) {
                if (!(isChar(a) && isChar(b) && (a == b || Math.abs(a - b) == 32)) && !(isNum(a) && isNum(b) && a == b)) {
                    return false;
                }
            }
            i++;
            j--;
        }
        return true;
    }

    private boolean isChar(char c) {
        return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z';
    }

    private boolean isNum(char c) {
        return c >= '0' && c <= '9';
    }

    public int maxProfit3(int[] prices) {
        int i1 = 0, i2 = 0, j1 = 0, j2 = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < prices[i1]) {
                i1 = i;
            } else if (prices[i] < prices[i2]) {
                i2 = i;
            }
            if (prices[i] > prices[j1]) {
                j1 = i;
            } else if (prices[i] > prices[j2]) {
                j2 = i;
            }
        }
        return prices[j1] + prices[j2] - prices[i1] - prices[i2];
    }

    public int maxProfit2(int[] prices) {
        int i, j = -1, sum = 0;
        for (i = 0; i < prices.length; i++) {
            if (j == -1 && (i == prices.length - 1 || prices[i] < prices[i + 1])) {
                j = i;
            } else if (j != -1 && prices[i] > prices[j] && (i == prices.length - 1 || prices[i] > prices[i + 1])) {
                sum += prices[i] - prices[j];
                j = -1;
            }
        }
        return sum;
    }

    public int maxProfit(int[] prices) {
        int min = -1, result = 0;
        for (int item : prices) {
            if (min == -1 || item < min) {
                min = item;
            } else {
                result = result > (item - min) ? result : (item - min);
            }
        }
        return result;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle.size() == 0) {
            return 0;
        }
        int[] dp = new int[triangle.size()];
        dp[0] = triangle.get(0).get(0);
        for (int i = 1; i < triangle.size(); i++) {
            List<Integer> list = triangle.get(i);
            dp[i] = dp[i - 1] + list.get(i);
            for (int j = i - 1; j >= 1; j--) {
                dp[j] = Math.min(dp[j - 1], dp[j]) + list.get(j);
            }
            dp[0] += list.get(0);
        }
        int min = dp[0];
        for (int item : dp) {
            if (item < min) {
                min = item;
            }
        }
        return min;
    }

    public List<Integer> getRow(int rowIndex) {
        List<Integer> oddList = new ArrayList<>();
        List<Integer> evenList = new ArrayList<>();
        oddList.add(1);
        oddList.add(1);
        evenList.add(1);
        for (int i = 2; i <= rowIndex; i++) {
            if (i % 2 == 0) {
                evenList.clear();
                evenList.add(1);
                for (int j = 0; j < oddList.size() - 1; j++) {
                    evenList.add(oddList.get(j) + oddList.get(j + 1));
                }
                evenList.add(1);
            } else {
                oddList.clear();
                oddList.add(1);
                for (int j = 0; j < evenList.size() - 1; j++) {
                    oddList.add(evenList.get(j) + evenList.get(j + 1));
                }
                oddList.add(1);
            }
        }
        if (rowIndex % 2 == 0) {
            return evenList;
        } else {
            return oddList;
        }
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new ArrayList<>();
        if (numRows == 0) {
            return result;
        }
        List<Integer> l1 = new ArrayList<>();
        l1.add(1);
        result.add(l1);
        if (numRows == 1) {
            return result;
        }
        List<Integer> l2 = new ArrayList<>();
        l2.add(1);
        l2.add(1);
        result.add(l2);
        if (numRows == 2) {
            return result;
        }
        for (int i = 2; i < numRows; i++) {
            List<Integer> l = new ArrayList<>();
            l.add(1);
            for (int j = 0; j < result.get(i - 1).size() - 1; j++) {
                l.add(result.get(i - 1).get(j) + result.get(i - 1).get(j + 1));
            }
            l.add(1);
            result.add(l);
        }
        return result;
    }

    public void connect(TreeLinkNode root) {
        TreeLinkNode dummyHead = new TreeLinkNode(0);
        TreeLinkNode pre = dummyHead;
        while (root != null) {
            if (root.left != null) {
                pre.next = root.left;
                pre = pre.next;
            }
            if (root.right != null) {
                pre.next = root.right;
                pre = pre.next;
            }
            root = root.next;
            if (root == null) {
                pre = dummyHead;
                root = dummyHead.next;
                dummyHead.next = null;
            }
        }
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        levelOrderBottomLoop(result, root, 0);
        return result;
    }

    private void levelOrderBottomLoop(List<List<Integer>> result, TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (depth >= result.size()) {
            result.add(0, new LinkedList<>());
        }
        result.get(result.size() - depth - 1).add(root.val);
        levelOrderBottomLoop(result, root.left, depth + 1);
        levelOrderBottomLoop(result, root.right, depth + 1);
    }

    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        if (inorder.length == 0) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postorder.length - 1]);
        buildTreeLoop2(inorder, postorder, root, 0, inorder.length - 1, 0, postorder.length - 1);
        return root;
    }

    private void buildTreeLoop2(int[] inorder, int[] postorder, TreeNode root, int il, int ir, int pl, int pr) {
        for (int i = il; i <= ir; i++) {
            if (inorder[i] == postorder[pr]) {
                root.left = root.right = null;
                if (i > il) {
                    root.left = new TreeNode(postorder[pl + i - il - 1]);
                    buildTreeLoop2(inorder, postorder, root.left, il, i - 1, pl, pl + i - il - 1);
                }
                if (i < ir) {
                    root.right = new TreeNode(postorder[pr - 1]);
                    buildTreeLoop2(inorder, postorder, root.right, i + 1, ir, pl + i - il, pr - 1);
                }
                break;
            }
        }
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length == 0) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[0]);
        buildTreeLoop(preorder, inorder, root, 0, preorder.length - 1, 0, inorder.length - 1);
        return root;
    }

    private void buildTreeLoop(int[] preorder, int[] inorder, TreeNode root, int pl, int pr, int il, int ir) {
        for (int i = il; i <= ir; i++) {
            if (inorder[i] == preorder[pl]) {
                root.left = root.right = null;
                if (i > il) {
                    root.left = new TreeNode(preorder[pl + 1]);
                    buildTreeLoop(preorder, inorder, root.left, pl + 1, pl + i - il, il, i - 1);
                }
                if (i < ir) {
                    root.right = new TreeNode(preorder[pl + i + 1 - il]);
                    buildTreeLoop(preorder, inorder, root.right, pl + i + 1 - il, pr, i + 1, ir);
                }
                break;
            }
        }
    }

    public int maxDepth(TreeNode root) {
        return root == null ? 0 : Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        zigzagLevelOrderLoop(result, root, 0);
        return result;
    }

    private void zigzagLevelOrderLoop(List<List<Integer>> result, TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (depth >= result.size()) {
            result.add(new LinkedList<>());
        }
        if (depth % 2 != 1) {
            result.get(depth).add(0, root.val);
        } else {
            result.get(depth).add(root.val);
        }
        zigzagLevelOrderLoop(result, root.left, depth + 1);
        zigzagLevelOrderLoop(result, root.right, depth + 1);
    }

    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        levelOrder2Loop(result, root, 0);
        return result;
    }

    private void levelOrder2Loop(List<List<Integer>> result, TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (depth >= result.size()) {
            result.add(new LinkedList<>());
        }
        result.get(depth).add(root.val);
        levelOrder2Loop(result, root.left, depth + 1);
        levelOrder2Loop(result, root.right, depth + 1);
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        List<List<Integer>> result = new ArrayList<>();
        List<TreeNode> temp = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        if (root != null) {
            queue.add(root);
            list.add(root.val);
            result.add(list);
        }
        while (true) {
            temp.clear();
            List<Integer> l = new ArrayList<>();
            while (!queue.isEmpty()) {
                TreeNode t = queue.poll();
                if (t.left != null) {
                    temp.add(t.left);
                    l.add(t.left.val);
                }
                if (t.right != null) {
                    temp.add(t.right);
                    l.add(t.right.val);
                }
            }
            if (temp.size() == 0) {
                break;
            }
            result.add(l);
            queue.addAll(temp);
        }
        return result;
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetricLoop(root.left, root.right);
    }

    private boolean isSymmetricLoop(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        } else if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return isSymmetricLoop(left.left, right.right) & isSymmetricLoop(left.right, right.left);
    }
}

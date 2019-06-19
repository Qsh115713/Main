package Code;

import Data.*;

import java.util.*;

public class ForOffer {

    public boolean isContinuous2(int[] numbers) {
        if (numbers == null || numbers.length == 0) return false;
        Arrays.sort(numbers);
        int num = 0;
        for (int i = 0; i < numbers.length; i++) {
            if (numbers[i] == 0) {
                ++num;
            } else if (i > 0 && numbers[i - 1] != 0) {
                int tmp = numbers[i] - numbers[i - 1] - 1;
                if (tmp < 0) return false;
                num -= tmp;
            }
        }
        return num >= 0;
    }

    public String ReverseSentence2(String str) {
        if (str.equals(" ")) return " ";
        if (str == null || str.length() <= 1) return str;
        StringBuilder res = new StringBuilder();
        int i, j = 0;
        boolean isWord = false;
        for (i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (c == ' ') {
                if (isWord) {
                    res.insert(0, str.substring(j, i)).insert(0, ' ');
                    isWord = false;
                }
            } else {
                if (!isWord) {
                    j = i;
                    isWord = true;
                }
            }
        }
        if (isWord) {
            res.insert(0, str.substring(j, i)).insert(0, ' ');
        }
        return res.length() == 0 ? "" : res.toString().substring(1);
    }

    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        ArrayList<Integer> res = new ArrayList<>();
        if (array.length < 2) return res;
        int lo = 0, hi = array.length - 1, curr;
        while (lo < hi) {
            curr = array[lo] + array[hi];
            if (curr == sum) {
                res.add(array[lo]);
                res.add(array[hi]);
                break;
            } else if (curr < sum) {
                ++lo;
            } else {
                --hi;
            }
        }
        return res;
    }

    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        int lo = 1, hi = 2, curr;
        while (lo < hi) {
            curr = (lo + hi) * (hi - lo + 1) / 2;
            if (curr == sum) {
                ArrayList<Integer> list = new ArrayList<>();
                for (int i = lo; i <= hi; i++) list.add(i);
                res.add(list);
                ++lo;
            } else if (curr < sum) {
                ++hi;
            } else {
                ++lo;
            }
        }
        return res;
    }

    public ArrayList<ArrayList<Integer>> FindContinuousSequence2(int sum) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        int lo = 1, hi = 2, mid = (1 + sum) / 2, num = lo + hi;
        while (lo < mid) {
            if (num == sum) {
                ArrayList<Integer> list = new ArrayList<>();
                for (int i = lo; i <= hi; i++) list.add(i);
                res.add(list);
            }
            while (lo < mid && sum < num) {
                num -= lo;
                ++lo;
                if (sum == num) {
                    ArrayList<Integer> list = new ArrayList<>();
                    for (int j = lo; j <= hi; ++j) list.add(j);
                    res.add(list);
                }
            }
            ++hi;
            num += hi;
        }
        return res;
    }

    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> res = new ArrayList<>();
        if (size == 0) return res;
        int begin;
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < num.length; i++) {
            begin = i - size + 1;
            if (q.isEmpty())
                q.add(i);
            else if (begin > q.peekFirst())
                q.pollFirst();

            while ((!q.isEmpty()) && num[q.peekLast()] <= num[i])
                q.pollLast();
            q.add(i);
            if (begin >= 0)
                res.add(num[q.peekFirst()]);
        }
        return res;
    }

    private Queue<Integer> p = new PriorityQueue<>();

    private Queue<Integer> q = new PriorityQueue<>();

    public void Insert(Integer num) {
        if (p.isEmpty() || num <= p.peek()) p.add(num);
        else q.add(num);
        if (p.size() == q.size() + 2) q.add(p.poll());
        if (p.size() + 1 == q.size()) p.add(q.poll());
    }

    public Double GetMedian() {
        if (p.isEmpty() && q.isEmpty()) return 0.0;
        return p.size() == q.size() ? (p.peek() + q.peek()) / 2.0 : p.peek() * 1.0;
    }

    public int[] multiply(int[] A) {
        if (A == null) return null;
        int[] B = new int[A.length];
        for (int i = 0; i < B.length; i++) {
            B[i] = 1;
        }
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B.length; j++) {
                if (i == j) continue;
                B[j] *= A[i];
            }
        }
        return B;
    }

    public boolean duplicate(int numbers[], int length, int[] duplication) {
        if (numbers == null) return false;
        Set<Integer> set = new HashSet<>();
        for (int number : numbers) {
            if (set.contains(number)) {
                duplication[0] = number;
                return true;
            } else {
                set.add(number);
            }
        }
        return false;
    }


    public int Sum_Solution(int n) {
        int res = n;
        boolean b = res != 0 && (res += Sum_Solution(n - 1)) != 0;
        return res;
    }

    public int LastRemaining_Solution(int n, int m) {
        if (n < 0 || m < 0) return -1;
        int prev = 0;
        for (int i = 2; i <= n; i++) {
            prev = (prev + m) % i;
        }
        return prev;
    }

    public boolean isContinuous(int[] numbers) {
        if (numbers.length == 0) return false;
        Arrays.sort(numbers);
        int i, index = 0, remain = numbers[0] == 0 ? 1 : 0, require = 0;
        for (i = 1; i < numbers.length; i++) {
            if (numbers[i] == 0) {
                remain++;
            } else {
                index = i;
                break;
            }
        }
        if (i >= numbers.length) return true;
        for (i = index + 1; i < numbers.length; i++) {
            if (numbers[i] == numbers[i - 1]) return false;
            require += numbers[i] - numbers[i - 1] - 1;
        }
        return require <= remain;
    }

    public String ReverseSentence(String str) {
        if (str.trim().equals("")) {
            return str;
        }
        String[] a = str.split(" ");
        StringBuffer o = new StringBuffer();
        int i;
        for (i = a.length; i > 0; i--) {
            o.append(a[i - 1]);
            if (i > 1) {
                o.append(" ");
            }
        }
        return o.toString();
    }

    public String LeftRotateString(String str, int n) {
        if (str == null || str.equals("")) return str;
        n %= str.length();
        StringBuilder sb = new StringBuilder();
        sb.append(str.substring(n)).append(str.substring(0, n));
        return sb.toString();
    }

    public void PrintToMaxOfNDigits(int n) {
        if (n <= 0) return;
        char[] chars = new char[n];
        for (int i = 0; i < 10; i++) {
            chars[0] = (char) ('0' + i);
            PrintToMaxOfNDigitsHelper(chars, 0);
        }
    }

    private void PrintToMaxOfNDigitsHelper(char[] chars, int index) {
        if (index == chars.length - 1) {
            System.out.println(chars);
            return;
        }
        for (int i = 0; i < 10; i++) {
            chars[index + 1] = (char) ('0' + i);
            PrintToMaxOfNDigitsHelper(chars, index + 1);
        }
    }

    public int FirstNotRepeatingChar(String str) {
        char c;
        int[] lower = new int[26];
        int[] index_lower = new int[26];
        int[] upper = new int[26];
        int[] index_upper = new int[26];
        for (int i = 0; i < str.length(); i++) {
            c = str.charAt(i);
            if (c <= 'Z') {
                upper[c - 'A']++;
                index_upper[c - 'A'] = i;
            } else {
                lower[c - 'a']++;
                index_lower[c - 'a'] = i;
            }
        }
        int index = str.length();
        for (int i = 0; i < lower.length; i++) {
            if (lower[i] == 1 && index > index_lower[i]) index = index_lower[i];
            if (upper[i] == 1 && index > index_upper[i]) index = index_upper[i];
        }
        return index == str.length() ? -1 : index;
    }

    public static int KMP(String ts, String ps) {
        char[] t = ts.toCharArray();
        char[] p = ps.toCharArray();
        int i = 0; // 主串的位置
        int j = 0; // 模式串的位置
        int[] next = getNext(ps);
        while (i < t.length && j < p.length) {
            if (j == -1 || t[i] == p[j]) { // 当j为-1时，要移动的是i，当然j也要归0
                i++;
                j++;
            } else {
                // i不需要回溯了
                // i = i - j + 1;
                j = next[j]; // j回到指定位置
            }
        }
        if (j == p.length) {
            return i - j;
        } else {
            return -1;
        }
    }

    public static int[] getNext(String ps) {
        char[] p = ps.toCharArray();
        int[] next = new int[p.length];
        next[0] = -1;
        int j = 0;
        int k = -1;
        while (j < p.length - 1) {
            if (k == -1 || p[j] == p[k]) {
                if (p[++j] == p[++k]) { // 当两个字符相等时要跳过
                    next[j] = next[k];
                } else {
                    next[j] = k;
                }
            } else {
                k = next[k];
            }
        }
        return next;

    }

    public int GetUglyNumber_Solution(int index) {
        if (index < 7) return index;
        int[] dp = new int[index];
        dp[0] = 1;
        int t2 = 0, t3 = 0, t5 = 0;
        for (int i = 1; i < index; i++) {
            dp[i] = Math.min(dp[t2] * 2, Math.min(dp[t3] * 3, dp[t5] * 5));
            if (dp[i] == dp[t2] * 2) t2++;
            if (dp[i] == dp[t3] * 3) t3++;
            if (dp[i] == dp[t5] * 5) t5++;
        }
        return dp[index - 1];
    }

    public String PrintMinNumber(int[] numbers) {
        String[] strs = new String[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            strs[i] = String.valueOf(numbers[i]);
        }
        Arrays.sort(strs, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String s1 = o1 + "" + o2;
                String s2 = o2 + "" + o1;
                return s1.compareTo(s2);
            }
        });
        StringBuilder sb = new StringBuilder();
        for (String str : strs) {
            sb.append(str);
        }
        return sb.toString();
    }

    public int FindGreatestSumOfSubArray(int[] array) {
        int res = array[0], cur = array[0];
        for (int i = 1; i < array.length; i++) {
            cur = Math.max(0, cur) + array[i];
            res = Math.max(res, cur);
        }
        return res;
    }

    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        if (k > input.length) return res;
        for (int i = (input.length - 1) / 2; i >= 0; i--) {
            sortHeap(input, i, input.length - 1);
        }
        for (int i = 0; i < k; i++) {
            res.add(input[0]);
            swap(input, 0, input.length - i - 1);
            sortHeap(input, 0, input.length - i - 2);
        }
        return res;
    }

    private void sortHeap(int[] input, int index, int end) {
        int left = 2 * index + 1, min = index;
        if (left <= end && input[left] < input[min]) min = left;
        if (left + 1 <= end && input[left + 1] < input[min]) min = left + 1;
        if (min == index) return;
        swap(input, index, min);
        sortHeap(input, min, end);
    }

    private void swap(int[] nums, int lo, int hi) {
        int t = nums[lo];
        nums[lo] = nums[hi];
        nums[hi] = t;
    }

    public int MoreThanHalfNum_Solution(int[] array) {
        if (array.length == 0) return 0;
        int res = array[0];
        int times = 1;
        for (int i = 1; i < array.length; i++) {
            if (times == 0) {
                res = array[i];
                times = 1;
            } else if (array[i] == res) times++;
            else times--;
        }
        if (times == 0) return 0;
        times = 0;
        for (int item : array) {
            if (item == res) times++;
        }
        return times >= (array.length + 1) / 2 ? res : 0;
    }

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null || pHead.next == null) return null;
        ListNode slow = pHead;
        ListNode fast = pHead;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) break;
        }
        if (fast == null) return null;
        fast = pHead;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    public ArrayList<String> Permutation(String str) {
        ArrayList<String> res = new ArrayList<>();
        if (str == null || str.equals("")) return res;
        char[] ch = str.toCharArray();
        permuteFunc(res, ch, 0);
        Collections.sort(res);
        return res;
    }

    private void permuteFunc(ArrayList<String> res, char[] ch, int index) {
        if (index == ch.length) {
            res.add(String.valueOf(ch));
            return;
        }
        Set<Character> charSet = new HashSet<>();
        for (int i = index; i < ch.length; i++) {
            if (i != index && charSet.contains(ch[i])) continue;
            charSet.add(ch[i]);
            swap(ch, i, index);
            permuteFunc(res, ch, index + 1);
            swap(ch, index, i);
        }
    }

    private void swap(char[] ch, int i, int j) {
        char c = ch[i];
        ch[i] = ch[j];
        ch[j] = c;
    }

    public TreeNode Convert2(TreeNode root) {
        if (root == null) return null;
        if (root.left == null && root.right == null) return root;
        TreeNode left = Convert2(root.left);
        TreeNode p = left;
        while (p != null && p.right != null) {
            p = p.right;
        }
        if (left != null) {
            p.right = root;
            root.left = p;
        }
        TreeNode right = Convert2(root.right);
        if (right != null) {
            root.right = right;
            right.left = root;
        }
        return left != null ? left : root;
    }

    public TreeNode Convert(TreeNode root) {
        Stack<TreeNode> s = new Stack<>();
        TreeNode p = root;
        TreeNode pre = null;
        while (p != null || !s.empty()) {
            while (p != null) {
                s.push(p);
                p = p.left;
            }
            p = s.pop();
            if (pre == null) {
                root = p;
                pre = root;
            } else {
                pre.right = p;
                p.left = pre;
                pre = p;
            }
            p = p.right;
        }
        return root;
    }

    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) return null;
        Map<RandomListNode, RandomListNode> map = new HashMap<>();
        RandomListNode qHead = new RandomListNode(pHead.label);
        map.put(pHead, qHead);
        RandomListNode p = pHead, q = qHead;
        while (p.next != null) {
            q.next = new RandomListNode(p.next.label);
            p = p.next;
            q = q.next;
            map.put(p, q);
        }
        p = pHead;
        q = qHead;
        while (p != null) {
            q.random = map.get(p.random);
            p = p.next;
            q = q.next;
        }
        return qHead;
    }

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        ArrayList<Integer> list = new ArrayList<>();
        list.add(root.val);
        travel(res, list, root, target, root.val);
        return res;
    }

    private void travel(ArrayList<ArrayList<Integer>> res, ArrayList<Integer> list, TreeNode root, int target, int count) {
        if (root.left == null && root.right == null) {
            if (count == target) res.add(new ArrayList<>(list));
            return;
        }
        if (root.left != null) {
            list.add(root.left.val);
            travel(res, list, root.left, target, count + root.left.val);
            list.remove(list.size() - 1);
        }
        if (root.right != null) {
            list.add(root.right.val);
            travel(res, list, root.right, target, count + root.right.val);
            list.remove(list.size() - 1);
        }
    }

    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence.length == 0) return false;
        return verify(sequence, 0, sequence.length - 1);
    }

    private boolean verify(int[] sequence, int il, int ir) {
        if (il > ir) return true;
        int i;
        for (i = il; i < ir; i++) if (sequence[i] > sequence[ir]) break;
        for (int j = i + 1; j < ir; j++) if (sequence[j] < sequence[ir]) return false;
        return verify(sequence, il, i - 1) && verify(sequence, i, ir - 1);
    }

    public int RectCover(int target) {
        if (target <= 2) return target;
        int pre1 = 1, pre2 = 2, res = 0;
        for (int i = 3; i <= target; i++) {
            res = pre1 + pre2;
            pre1 = pre2;
            pre2 = res;
        }
        return res;
    }

    public int JumpFloorII(int target) {
        return target == 0 ? 0 : (int) Math.pow(2, target - 1);
    }

    public int JumpFloor(int target) {
        int[] dp = new int[target + 1];
        dp[1] = 1;
        if (target <= 1) return dp[target];
        dp[2] = 2;
        for (int i = 3; i <= target; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[target];
    }

    public int Fibonacci(int n) {
        int[] dp = new int[40];
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int minNumberInRotateArray(int[] array) {
        if (array.length == 0) return 0;
        int n = array.length, lo = 0, hi = n - 1, mid;
        if (array[lo] <= array[hi]) return array[lo];
        while (lo <= hi) {
            mid = lo + (hi - lo) / 2;
            if (mid != 0 && array[mid - 1] > array[mid]) return array[mid];
            if (mid != n - 1 && array[mid] > array[mid + 1]) return array[mid + 1];
            if (array[mid] == array[lo]) return Math.min(array[lo], array[hi]);
            if (array[mid] > array[lo]) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return 0;
    }

    private Stack<Integer> stack1 = new Stack<Integer>();
    private Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (!stack2.empty()) {
            return stack2.pop();
        } else {
            while (!stack1.empty()) {
                stack2.push(stack1.pop());
            }
            return stack2.empty() ? -1 : stack2.pop();
        }
    }

    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length != in.length) return null;
        return reConstructBinaryTreeHelper(pre, in, 0, pre.length - 1, 0, in.length - 1);
    }

    private TreeNode reConstructBinaryTreeHelper(int[] pre, int[] in, int pl, int pr, int il, int ir) {
        if (pl > pr || il > ir) return null;
        TreeNode root = new TreeNode(pre[pl]);
        int i;
        for (i = il; i <= ir; i++) {
            if (in[i] == pre[pl]) break;
        }
        root.left = reConstructBinaryTreeHelper(pre, in, pl + 1, i - il + pl, il, i - 1);
        root.right = reConstructBinaryTreeHelper(pre, in, i - il + pl + 1, pr, i + 1, ir);
        return root;
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> res = new ArrayList<>();
        ListNode p = listNode;
        while (p != null) {
            res.add(0, p.val);
            p = p.next;
        }
        return res;
    }

    public String replaceSpace(StringBuffer str) {
        int m = 0, n = str.length();
        for (int i = 0; i < n; ++i) if (str.charAt(i) == ' ') ++m;
        str.setLength(n + 2 * m);
        int i = n - 1, j = n + 2 * m - 1;
        while (i >= 0 && i < j) {
            char c = str.charAt(i);
            if (c == ' ') {
                str.replace(j - 2, j + 1, "%20");
                j -= 3;
            } else {
                str.replace(j, j + 1, "" + c);
                --j;
            }
            --i;
        }
        return str.toString();
    }

    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0 || array[0].length == 0) return false;
        int row = 0, col = array[0].length - 1;
        while (row <= array.length - 1 && col >= 0) {
            if (array[row][col] == target) return true;
            else if (array[row][col] > target) --col;
            else ++row;
        }
        return false;
    }
}

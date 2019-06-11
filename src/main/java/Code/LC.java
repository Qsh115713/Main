package Code;

import Data.*;

import java.util.*;

public class LC {

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
        for (int i = 0; i < strs.length; i++) {
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
        StringBuffer sb = new StringBuffer();
        for (String str : strs) {
            sb.append(str);
        }
        return sb.toString();
    }

    public int FindGreatestSumOfSubArray(int[] array) {
        if (array.length == 0) return 0;
        int res = Integer.MIN_VALUE, sum = 0;
        for (int num : array) {
            sum += num;
            if (sum > res) {
                res = sum;
            }
            if (sum < 0) {
                sum = 0;
            }
        }
        return res;
    }

    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        if (k > input.length) return new ArrayList<>();
        PriorityQueue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
        for (int num : input) {
            q.add(num);
            if (q.size() > k) {
                q.poll();
            }
        }
        return new ArrayList<>(q);
    }

    public int MoreThanHalfNum_Solution(int[] array) {
        if (array.length == 0) return 0;
        int res = 0, count = 0;
        for (int num : array) {
            if (count == 0) {
                res = num;
                count++;
            } else {
                if (res == num) {
                    count++;
                } else {
                    count--;
                }
            }
        }
        if (count == 0) return 0;
        count = 0;
        for (int num : array) {
            if (num == res) ++count;
        }
        return count >= (array.length + 1) / 2 ? res : 0;
    }

    public int JumpFloorII(int target) {
        return (int) Math.pow(2, target - 1);
    }

    public int JumpFloor(int target) {
        return Fibonacci(target - 1);
    }

    public int Fibonacci(int n) {
        if (n <= 1) return n;
        int pre = 0, p = 1, now = 1;
        for (int i = 2; i <= n; i++) {
            now = pre + p;
            pre = p;
            p = now;
        }
        return now;
    }

    public int minNumberInRotateArray(int[] array) {
        if (array.length == 0) return 0;
        int n = array.length, lo = 0, hi = n - 1, mid;
        if (array[lo] <= array[hi]) return array[lo];
        while (lo <= hi) {
            mid = (lo + hi) >> 1;
            if (mid != 0 && array[mid - 1] > array[mid]) return array[mid];
            if (mid != n - 1 && array[mid] > array[mid + 1]) return array[mid + 1];
            if (array[mid] == array[lo]) return Math.min(array[lo], array[hi]);
            if (array[mid] > array[lo]) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }

    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (stack2.empty()) {
            while (!stack1.empty()) {
                stack2.push(stack1.pop());
            }
        }
        return stack2.empty() ? -1 : stack2.pop();
    }

    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length != in.length) return null;
        return reConstructBinaryTreeHelper(pre, in, 0, pre.length - 1, 0, in.length - 1);
    }

    private TreeNode reConstructBinaryTreeHelper(int[] pre, int[] in, int pl, int pr, int il, int ir) {
        if (pl > pr || il > ir) return null;
        TreeNode root = new TreeNode(pre[pl]);
        int i = il;
        for (; i < ir; i++) {
            if (in[i] == pre[pl]) break;
        }
        root.left = reConstructBinaryTreeHelper(pre, in, pl + 1, i + pl - il, il, i - 1);
        root.right = reConstructBinaryTreeHelper(pre, in, i + pl - il + 1, pr, i + 1, ir);
        return root;
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> res = new ArrayList<>();
        while (listNode != null) {
            res.add(0, listNode.val);
            listNode = listNode.next;
        }
        return res;
    }

    public String replaceSpace(StringBuffer str) {
        int m = 0, n = str.length();
        for (int i = 0; i < n; i++) {
            if (str.charAt(i) == ' ') {
                ++m;
            }
        }
        str.setLength(n + 2 * m);
        int i = n - 1, j = str.length() - 1;
        while (i >= 0 && i < j) {
            char c = str.charAt(i);
            if (c == ' ') {
                str.replace(j - 2, j + 1, "%20");
                j -= 3;
            } else {
                str.replace(j, j + 1, "" + c);
                j--;
            }
            i--;
        }
        return str.toString();
    }

    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0 || array[0].length == 0) return false;
        int m = array.length, n = array[0].length, i = 0, j = n - 1;
        while (i < m && j >= 0) {
            if (target == array[i][j]) {
                return true;
            } else if (target < array[i][j]) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }
}

package Review;

import Data.*;

import java.util.*;

public class ForOffer {

    public int movingCount(int threshold, int rows, int cols) {
        boolean[][] flag = new boolean[rows][cols];
        return movingCountHelper(flag, threshold, rows, cols, 0);
    }

    private int movingCountHelper(boolean[][] flag, int threshold, int rows, int cols, int index) {
        int i = index / cols, j = index % cols;
        if (i < 0 || i >= rows || j < 0 || j >= cols || flag[i][j] || getSplitVal(i) + getSplitVal(j) > threshold)
            return 0;
        flag[i][j] = true;
        return movingCountHelper(flag, threshold, rows, cols, index - cols)
                + movingCountHelper(flag, threshold, rows, cols, index + cols)
                + movingCountHelper(flag, threshold, rows, cols, index - 1)
                + movingCountHelper(flag, threshold, rows, cols, index + 1) + 1;
    }

    private int getSplitVal(int num) {
        int res = 0;
        while (num != 0) {
            res += num % 10;
            num /= 10;
        }
        return res;
    }

    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        boolean[] flag = new boolean[matrix.length];
        for (int i = 0; i < matrix.length; i++)
            if (hasPathHelper(matrix, rows, cols, str, flag, i, 0)) return true;
        return false;
    }

    private boolean hasPathHelper(char[] matrix, int rows, int cols, char[] str, boolean[] flag, int mIndex, int sIndex) {
        if (sIndex == str.length) return true;
        int i = mIndex / cols, j = mIndex % cols;
        if (i < 0 || i >= rows || j < 0 || j >= cols || matrix[mIndex] != str[sIndex] || flag[mIndex]) return false;
        flag[mIndex] = true;
        if (hasPathHelper(matrix, rows, cols, str, flag, mIndex - cols, sIndex + 1)
                || hasPathHelper(matrix, rows, cols, str, flag, mIndex + cols, sIndex + 1)
                || hasPathHelper(matrix, rows, cols, str, flag, mIndex - 1, sIndex + 1)
                || hasPathHelper(matrix, rows, cols, str, flag, mIndex + 1, sIndex + 1)) return true;
        flag[mIndex] = false;
        return false;
    }

    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> res = new ArrayList<>();
        if (size == 0) return res;
        int begin;
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < num.length; i++) {
            begin = i - size + 1;
            if (q.isEmpty()) q.add(i);
            else if (begin > q.peekFirst()) q.pollFirst();
            while (!q.isEmpty() && num[q.peekLast()] <= num[i]) q.pollLast();
            q.add(i);
            if (begin >= 0) res.add(num[q.peekFirst()]);
        }
        return res;
    }

    private PriorityQueue<Integer> largeHeap = new PriorityQueue<>(Comparator.reverseOrder());
    private PriorityQueue<Integer> smallHeap = new PriorityQueue<>();

    public void Insert(Integer num) {
        if (largeHeap.size() > smallHeap.size()) {
            if (num < largeHeap.peek()) {
                int tmp = largeHeap.poll();
                largeHeap.add(num);
                smallHeap.add(tmp);
            } else {
                smallHeap.add(num);
            }
        } else {
            if (smallHeap.size() == 0) {
                largeHeap.add(num);
                return;
            }
            if (num > smallHeap.peek()) {
                int tmp = smallHeap.poll();
                smallHeap.add(num);
                largeHeap.add(tmp);
            } else {
                largeHeap.add(num);
            }
        }
    }

    public Double GetMedian() {
        if (largeHeap.size() == 0) return 0.0;
        if (largeHeap.size() > smallHeap.size()) {
            return 1.0 * largeHeap.peek();
        } else {
            int a = largeHeap.peek();
            int b = smallHeap.peek();
            return (a + b) / 2.0;
        }
    }

    ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (pRoot == null) return res;
        Queue<TreeNode> q = new ArrayDeque<>();
        q.add(pRoot);
        while (!q.isEmpty()) {
            ArrayList<Integer> list = new ArrayList<>();
            int lo = 0, hi = q.size();
            while (lo++ < hi) {
                TreeNode tmp = q.poll();
                list.add(tmp.val);
                if (tmp.left != null) q.add(tmp.left);
                if (tmp.right != null) q.add(tmp.right);
            }
            res.add(list);
        }
        return res;
    }

    ArrayList<ArrayList<Integer>> Print2(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        PrintHelper(res, pRoot, 0);
        return res;
    }

    void PrintHelper(ArrayList<ArrayList<Integer>> res, TreeNode pRoot, int depth) {
        if (pRoot == null) return;
        if (depth >= res.size()) {
            res.add(new ArrayList<>());
        }
        PrintHelper(res, pRoot.left, depth + 1);
        PrintHelper(res, pRoot.right, depth + 1);
        res.get(depth).add(pRoot.val);
    }
}

import Code.*;

import java.util.*;


public class Main {


    public static void main(String[] args) {
        LeetCode lc = new LeetCode();

        Scanner sc = new Scanner(System.in);
        int m = sc.nextInt();
        int n = sc.nextInt();
        int[] w = new int[n + 1];
        int[] v = new int[n + 1];
        int[] c = new int[n+1];
        for (int i = 1; i <= n; i++) {
            w[i] = sc.nextInt();
            v[i] = sc.nextInt();
            c[i] = sc.nextInt();
        }

        System.out.println(lc.multiBag(w, v, c, m));

        /*TreeNode s = new TreeNode(3);
        s.left = new TreeNode(4);
        s.right = new TreeNode(5);
        s.left.left = new TreeNode(1);
        s.left.right = new TreeNode(2);

        TreeNode t = new TreeNode(4);
        t.left = new TreeNode(1);
        t.right = new TreeNode(2);

        System.out.println(lc.isSubtree(s,t));*/

       /* ListNode l1 = new ListNode(1);
        ListNode l2 = new ListNode(2);
        ListNode l3 = new ListNode(3);
        ListNode l4 = new ListNode(4);
        ListNode l5 = new ListNode(5);
        ListNode l6 = new ListNode(6);
        ListNode l7 = new ListNode(7);
        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        l4.next = l5;
        l5.next = l6;
        l6.next = l7;
        l7.next = null;


        ListNode l = l7;
        if (l.next==null) {
            l = null;
        }*/

        //System.out.println();

        /*
        Scanner cin = new Scanner(System.in);
        while (cin.hasNext()) {
            String[] pwd = new String[3];
            for (int i = 0; i < 3; i++) {
                pwd[i] = cin.nextLine();
            }
            if (pwd[0].charAt(0) == pwd[2].charAt(2) && pwd[0].charAt(1) == pwd[2].charAt(1)
                    && pwd[0].charAt(2) == pwd[2].charAt(0) && pwd[1].charAt(0) == pwd[1].charAt(2)) {
                System.out.println("YES");
            } else {
                System.out.println("NO");
            }
        }*/
    }
}

/*public class Main {



    public static void main(String[] args) {
        Review101_200 lc = new Review101_200();
        *//*TreeNode root = new TreeNode(7);
        root.left = new TreeNode(3);
        root.right = new TreeNode(9);
        root.left.left = new TreeNode(2);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(8);
        root.right.right = new TreeNode(11);
        BSTIterator iterator = new BSTIterator(null);
        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }*//*

        //int[] nums = new int[] {4704,6306,9385,7536,3462,4798,5422,5529,8070,6241,9094,7846,663,6221,216,6758,8353,3650,3836,8183,3516,5909,6744,1548,5712,2281,3664,7100,6698,7321,4980,8937,3163,5784,3298,9890,1090,7605,1380,1147,1495,3699,9448,5208,9456,3846,3567,6856,2000,3575,7205,2697,5972,7471,1763,1143,1417,6038,2313,6554,9026,8107,9827,7982,9685,3905,8939,1048,282,7423,6327,2970,4453,5460,3399,9533,914,3932,192,3084,6806,273,4283,2060,5682,2,2362,4812,7032,810,2465,6511,213,2362,3021,2745,3636,6265,1518,8398};
        *//*int[] nums = new int[] {2000,216,2,2060,213};
        System.out.println(lc.largestNumber(nums));*//*
 *//*ListNode l1 = new ListNode(1);
        ListNode l2 = new ListNode(2);
        ListNode l3 = new ListNode(3);
        ListNode l4 = new ListNode(4);
        ListNode l5 = new ListNode(5);
        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        l4.next = l5;
        l5.next = null;*//*

 *//*char[][] board = new char[][] {{'5','3','.','.','7','.','.','.','.'},{'6','.','.','1','9','5','.','.','.'},{'.','9','8','.','.','.','.','6','.'},{'8','.','.','.','6','.','.','.','3'},{'4','.','.','8','.','3','.','.','1'},{'7','.','.','.','2','.','.','.','6'},{'.','6','.','.','.','.','2','8','.'},{'.','.','.','4','1','9','.','.','5'},{'.','.','.','.','8','.','.','7','9'}};
        lc.solveSudoku(board);
        for (int i=0;i<9;i++) {
            for (int j=0;j<9;j++) {
                System.out.print(board[i][j] + "\t");
            }
            System.out.println();
        }
        List<Interval> intervals = new ArrayList<>();
        Interval interval = new Interval(6, 8);
        Interval interval1 = new Interval(1, 5);
        Interval interval2 = new Interval(3, 5);
        Interval interval3 = new Interval(6, 7);
        Interval interval4 = new Interval(8, 10);
        Interval interval5 = new Interval(12, 16);
        intervals.add(interval1);
        intervals.add(interval2);
        intervals.add(interval3);
        intervals.add(interval4);
        intervals.add(interval1);
        lc.insert(intervals, interval);*//*

 *//*char[][] board =
                {
                        {'A', 'B'},
                        {'C', 'D'},
                };
        System.out.println(lc.numTrees2(4));*//*

        TreeNode root = new TreeNode(3);
        root.left = new TreeNode(9);
        root.right = new TreeNode(20);
        *//*root.left.left = new TreeNode(1);
        root.left.right = new TreeNode(3);*//*
        root.right.left = new TreeNode(15);
        root.right.right = new TreeNode(7);




        *//*List<String> wordList = new ArrayList<>();
        wordList.add("hot");
        wordList.add("dot");
        wordList.add("dog");
        wordList.add("lot");
        wordList.add("log");
        wordList.add("cog");
        lc.findLadders("hit", "cog", wordList);*//*

        int[] gas = {1, 2, 3, 4, 5};
        int[] cost = {3, 4, 5, 1, 2};
        lc.canCompleteCircuit(gas, cost);
    }
}*/

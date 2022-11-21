package sort;

public class Solution {

    // 斐波那契数 dp[i]=dp[i-1]+dp[i-2]
    public static int fib(int n) {
        // 特殊情况
        if (n <= 1) return n;
        // 确定数组及其下标的含义
        int dp[] = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 2] + dp[i - 1];
        }
        return dp[n];
    }
    // 爬楼梯
    public static int climbStairs(int n) {
        int sum=0;  //总共有几种爬法
        if(n<=2) return n;
        else{
            int first=1;    //再补爬两个楼梯
            int second=2;   //再补爬一个楼梯
            for(int i=0;i<n-2;i++){
                sum=first+second;
                first=second;
                second=sum;
            }
        }
        return sum;
    }

    public static void main(String[] args) {
//        // 斐波那契数
//        int fibN = fib(10);
//        System.out.println("斐波那契数结果:" + fibN);
        // 爬楼梯
        int climbStairsN = climbStairs(10000);
        System.out.println("爬楼梯结果:" + climbStairsN);
    }
}

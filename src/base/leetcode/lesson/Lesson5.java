package lesson;

/**
 * 排列硬币
 */
public class Lesson5 {

    /**
     * 解法1：迭代
     */
    public static int arrangeCoins(int n) {
        for(int i=1; i<=n; i++){
            n = n-i;
            if (n <= i){
                return i;
            }
        }
        return 0;
    }



    /**
     * 解法2：二分查找
     */
    public static int arrangeCoins2(int n) {
        int low = 0, high = n;
        while (low <= high) {
            long mid = (high - low) / 2 + low;
            long cost = ((mid + 1) * mid) / 2;
            if (cost == n) {
                return (int)mid;
            } else if (cost > n) {
                high = (int)mid - 1;
            } else {
                low = (int)mid + 1;
            }
        }
        return high;
    }

    /**
     * 解法3：牛顿迭代
     */
    public static int newton(int x) {
        if(x==0) return 0;
        return ((int)(sqrts(x,x)));
    }

    public static double sqrts(double x,int n){
        double res = (x + (2*n-x) / x) / 2;
        if (res == x) {
            return x;
        } else {
            return sqrts(res,n);
        }
    }



    public static void main(String[] args) {
        long ts0 = System.currentTimeMillis();
        System.out.println("meth 1 = " + arrangeCoins(300));
        long ts1 = System.currentTimeMillis();
        System.out.println("time 1 = " + (ts1 - ts0) + "ms");
        System.out.println("meth 2 = " + arrangeCoins2(300));
        long ts2 = System.currentTimeMillis();
        System.out.println("time 2 = " + (ts2 - ts1) + "ms");
        System.out.println("meth 3 = " + newton(300));
        long ts3 = System.currentTimeMillis();
        System.out.println("time 3 = " + (ts3 - ts2) + "ms");
    }
}

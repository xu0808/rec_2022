package lesson;

/**
 * x的平方根
 */
public class Lesson3 {

    /**
     * 解法1：二分查找
     */
    public static int binarySearch(int x) {
        int l = 0, r = x, index = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if ((long) mid * mid <= x) {
                index = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return index;
    }


    /**
     * 解法2：牛顿迭代
     */
    public static int newton(int x) {
        if(x==0) return 0;
        return ((int)(sqrts(x,x)));
    }
    public static double sqrts(double i,int x){
        double res = (i + x / i) / 2;
        if (res == i) {
            return i;
        } else {
            return sqrts(res,x);
        }
    }


    /**
     * 解法3：梯度下降
     */
    public static int grad(int x) {
        double r = x;
        while (r*r - x > 0.00001 ) {
           r = r - 0.05 *r;
            System.out.println("r = " + r);
        }
        return (int)r;
    }


    public static void main(String[] args) {
        System.out.println("meth 1 = " + binarySearch(222));
        System.out.println("meth 2 = " + newton(222 ));
        System.out.println("meth 3 = " + grad(222 ));
    }
}

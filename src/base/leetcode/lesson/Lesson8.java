package lesson;

import java.util.Arrays;

/**
 * 三角形的最大周长
 * 给定由一些正数（代表长度）组成的数组 A ，返回由其中三个长度组成的、面积不为零的三角形的最大
 * 周长。
 */
public class Lesson8 {

    public static int largestPerimeter(int[] A) {
        Arrays.sort(A);
        for (int i = A.length - 1; i >= 2; --i) {
            if (A[i - 2] + A[i - 1] > A[i]) {
                return A[i - 2] + A[i - 1] + A[i];
            }
        }
        return 0;
    }
    public static void main(String[] args) {
        long ts0 = System.currentTimeMillis();
        System.out.println("meth 1 = " + largestPerimeter(new int[]{5,18,20}));
        long ts1 = System.currentTimeMillis();
        System.out.println("time 1 = " + (ts1 - ts0) + "ms");
    }
}

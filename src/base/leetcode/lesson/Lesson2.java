package lesson;

/**
 * 统计N以内的素数
 */
public class Lesson2 {

    /**
     * 解法1：暴力算法
     */
    public static int countPrimes(int n) {
        int ans = 0;
        for (int i = 2; i < n; ++i) {
            ans += isPrime(i) ? 1 : 0;
        }
        return ans;
    }
    //i如果能被x整除，则x/i肯定能被x整除，因此只需判断i和根号x之中较小的即可
    public static boolean isPrime(int x) {
        for (int i = 2; i * i <= x; ++i) {
            if (x % i == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * 解法2：埃氏筛
     * 利用合数的概念(非素数)，素数*n必然是合数，因此可以从2开始遍历，将所有的合数做上标记
     */
    public static int eratosthenes(int n) {
        boolean[] isPrime = new boolean[n];
        int ans = 0;
        for (int i = 2; i < n; i++) {
            if (!isPrime[i]) {
                ans += 1;
                for (int j = i * i; j < n; j += i) {
                    isPrime[j] = true;
                }
            }
        }
        return ans;
    }

    public static void main(String[] args) {
        System.out.println("meth 1 = " + countPrimes(53));
        System.out.println("meth 2 = " + eratosthenes(53 ));
    }

}

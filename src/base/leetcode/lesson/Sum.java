package lesson;

public class Sum {
    public static void main(String[] args) {
        int[] arr = new int[]{14055,
                42762,
                45034,
                49245,
                67466,
                32613,
                32597,
                38035,
                199831
        };
        // 521638
        int sum = 0;
        for (int i : arr) {
            sum += i;
        }
        System.out.println("sum = " + sum);
        // 源石31 %
        double up = (72.0 - 55) / 55.0;
        System.out.println("up = " + String.format("%.6f", up));
    }
}

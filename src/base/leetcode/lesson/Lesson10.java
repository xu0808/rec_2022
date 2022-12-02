package lesson;

/**
 * 预测赢家
 */
public class Lesson10 {

    /**
     * 解法1：迭代
     */
    public static int maxScore(int[] nums, int l, int r) {
        // 剩下一个值，只能取该值
        if (l == r) {
            return nums[l];
        }

        int selectLeft = 0, selectRight = 0;

        // 剩下两个值，取较大的
        if (r - l == 1) {
            selectLeft = nums[l];
            selectRight = nums[r];
        }

        // 剩下大于两个值，先手选一边(使自己得分最高的一边)，后手则选使对手得分最低的一边
        if (r - l >= 2) {
            selectLeft = nums[l] + Math.min(maxScore(nums, l + 2, r), maxScore(nums,l + 1, r - 1));
            selectRight = nums[r] + Math.min(maxScore(nums, l + 1, r - 1), maxScore(nums, l, r - 2));
        }

        return Math.max(selectLeft, selectRight);
    }

    public static void main(String[] args) {
        long ts0 = System.currentTimeMillis();
        // 偶数个存在必赢方法
        // int[] arr = (new int[]{5, 200, 1, 6});
        // 第二值超级大，保证必须输
        int[] arr = (new int[]{5, 200, 1, 3, 6});
        int sum = 0;
        for(int i:arr){
            sum += i;
        }
        int p1 = maxScore(arr, 0, arr.length -1);
        int p2 = sum - p1;
        System.out.println("p1 = " + p1 + ",p2 = " + p2 + ",p1 >= p2 == " + (p1 >= p2));
        long ts1 = System.currentTimeMillis();
        System.out.println("time 1 = " + (ts1 - ts0) + "ms");
    }
}

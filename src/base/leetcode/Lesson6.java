/**
 * 柠檬水找零
 */
public class Lesson6 {

    public static boolean lemonadeChange(int[] bills) {
        int fiveNum = 0, tenNum = 0;
        for (int bill : bills) {
            // 1、5元直接添加
            if (bill == 5) {
                fiveNum++;
            // 2、10元需要验证是否存在5元的
            } else if (bill == 10) {
                if (fiveNum == 0) {
                    return false;
                }
                fiveNum--;
                tenNum++;
            // 3、20元需要验证是否存在5、10元的足够找零（优先找零10元的）
            } else {
                if (fiveNum > 0 && tenNum > 0) {
                    fiveNum--;
                    tenNum--;
                } else if (fiveNum >= 3) {
                    fiveNum -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }



    public static void main(String[] args) {
        long ts0 = System.currentTimeMillis();
        System.out.println("meth 1 = " + lemonadeChange(new int[]{5,5,5,10,20}));
        long ts1 = System.currentTimeMillis();
        System.out.println("time 1 = " + (ts1 - ts0) + "ms");
    }
}

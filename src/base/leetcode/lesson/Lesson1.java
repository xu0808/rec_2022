package lesson;

/**
 * 第1课 反转链表
 * 输入: 1->2->3->4->5
 * 输出: 5->4->3->2->1
 */
public class Lesson1 {

    static class ListNode {
        int val;
        ListNode next;

        public ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    /**
     * 解法1：迭代
     * 重复某一过程，每一次处理结果作为下一次处理的初始值，这些初始值类似于状态、每次处理都会改变状态、直至到达最终状态
     */
    public static ListNode iterate(ListNode head) {
        ListNode prev = null, curr, next;
        curr = head;
        while (curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    /**
     * 解法2：递归：
     * 理解方式：先递归最末端查看
     * 所以层都返回新head，并反转当前指针
     */
    public static ListNode reverse(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverse(head.next);
        // 反转next指针
        head.next.next = head;
        // 删除当前next指针
        head.next = null;
        return newHead;
    }

    public static void main(String[] args) {
        ListNode node5 = new ListNode(5, null);
        ListNode node4 = new ListNode(4, node5);
        ListNode node3 = new ListNode(3, node4);
        ListNode node2 = new ListNode(2, node3);
        ListNode node1 = new ListNode(1, node2);
        //ListNode node = iterate(node1);
        ListNode node_1 = reverse(node1);
        System.out.println(node_1);
    }

}

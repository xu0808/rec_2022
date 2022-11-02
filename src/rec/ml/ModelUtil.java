package com.xxyp.api.common.utils;


import lombok.extern.slf4j.Slf4j;


import java.util.HashMap;
import java.util.Map;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.*;
import java.io.*;
import java.util.*;
import java.util.List;

/**
 *     <dependency>
 *             <groupId>org.jpmml</groupId>
 *             <artifactId>pmml-evaluator</artifactId>
 *             <version>1.5.15</version>
 *         </dependency>
 *         <dependency>
 *             <groupId>com.sun.xml.bind</groupId>
 *             <artifactId>jaxb-impl</artifactId>
 *             <version>2.1.2</version>
 *         </dependency>
 *         <dependency>
 *             <groupId>javax.xml.bind</groupId>
 *             <artifactId>jaxb-api</artifactId>
 *             <version>2.3.0</version>
 *         </dependency>
 *         <!-- https://mvnrepository.com/artifact/javax.activation/javax.activation -->
 *         <dependency>
 *             <groupId>javax.activation</groupId>
 *             <artifactId>activation</artifactId>
 *             <version>1.1.1</version>
 *         </dependency>
 *
 */


@Slf4j
public class ModelUtil {

    public static void main(String args[]) throws Exception {
        //模型路径
        String modelFile = "C:\\work\\data\\rec\\ml\\GBDT+LR.pmml";
        //传入模型特征数据
        // cols: ['Age', 'Employment', 'Education', 'Marital', 'Occupation', 'Income', 'Gender', 'Deductions', 'Hours', 'Adjusted']
        // first line: [38 'Private' 'College' 'Unmarried' 'Service' 81838.0 'Female' False 72 0]
        String cols = "Age,Employment,Education,Marital,Occupation,Income,Gender,Deductions,Hours,Adjusted";
        String values = "38,Private,College,Unmarried,Service,81838.0,Female,False,72,0";
        Map<String, Object> data = new HashMap();
        for(int i=0;i<cols.split(",").length;i++){
            data.put(cols.split(",")[i], values.split(",")[i]);
        }

        ModelUtil obj = new ModelUtil();
        Evaluator model = new LoadingModelEvaluatorBuilder()
                .load(new File(modelFile))
                .build();

        Map<String, Object> output = obj.predict(model, data);
        System.out.println("X=" + data + " -> y=" + output.get("Adjusted"));
    }


    /**
     * 运行模型得到结果。
     */
    private Map<String, Object> predict(Evaluator evaluator, Map<String, Object> data) {
        Map<FieldName, FieldValue> input = getFieldMap(evaluator, data);
        Map<String, Object> output = evaluate(evaluator, input);
        return output;
    }

    /**
     * 把原始输入转换成PMML格式的输入。
     */
    private Map<FieldName, FieldValue> getFieldMap(Evaluator evaluator, Map<String, Object> input) {
        List<InputField> inputFields = evaluator.getInputFields();
        Map<FieldName, FieldValue> map = new LinkedHashMap<FieldName, FieldValue>();
        for (InputField field : inputFields) {
            FieldName fieldName = field.getName();
            Object rawValue = input.get(fieldName.getValue());
            FieldValue value = field.prepare(rawValue);
            map.put(fieldName, value);
        }
        return map;
    }

    /**
     * 运行模型得到结果。
     */
    private Map<String, Object> evaluate(Evaluator evaluator, Map<FieldName, FieldValue> input) {
        Map<FieldName, ?> results = evaluator.evaluate(input);
        List<TargetField> targetFields = evaluator.getTargetFields();
        Map<String, Object> output = new LinkedHashMap<String, Object>();
        for (int i = 0; i < targetFields.size(); i++) {
            TargetField field = targetFields.get(i);
            FieldName fieldName = field.getName();
            Object value = results.get(fieldName);
            if (value instanceof Computable) {
                Computable computable = (Computable) value;
                value = computable.getResult();
            }
            output.put(fieldName.getValue(), value);
        }
        return output;
    }
}

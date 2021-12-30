    /* class name MUST BE THE SAME AS THE FILE NAME */
/*Java is capital-sensitive*/
import java.lang.Math; /*imports java's math library*/
import java.util.Random; //imports random library

import java.util.Arrays; //lets you use arrays easily
import java.lang.Integer;
import java.lang.String;
import java.lang.Double;
import java.io.*;

import java.lang.Thread;

class java_generative_adverserial_network
{



  
  public static double[][] init(int layers[], double[] initializing_range)
  {
   int weights_amount=0;
   double[][] outputs_of_init={{},{},{},{}};
   double[] nodes_counted_in_each_layer={0, layers[0]};
    for (int i=0; i<layers.length-1; i++)
    {
    weights_amount+=layers[i]*layers[i+1];
    }
    for (int i1=0; i1<weights_amount; i1++)
    {
      outputs_of_init[0]=Arrays.copyOf(outputs_of_init[0], outputs_of_init[0].length+1);
      outputs_of_init[0][outputs_of_init[0].length-1]=((double)((Math.random() * (initializing_range[1] - initializing_range[0])) + initializing_range[0]));
    }

    int biases_amount=0;
    for (int i2=0; i2<layers.length-1; i2++)
    {
      biases_amount+=layers[i2+1];
    }
    for (int i3=0; i3<biases_amount; i3++)
    {
      outputs_of_init[1]=Arrays.copyOf(outputs_of_init[1], outputs_of_init[1].length+1);
      outputs_of_init[1][outputs_of_init[1].length-1]=((double)((Math.random() * (initializing_range[1] - initializing_range[0])) + initializing_range[0]));
    }

    outputs_of_init[2]=Arrays.copyOf(outputs_of_init[2], outputs_of_init[2].length+1);
    outputs_of_init[2][outputs_of_init[2].length-1]=weights_amount;
    outputs_of_init[2]=Arrays.copyOf(outputs_of_init[2], outputs_of_init[2].length+1);
    outputs_of_init[2][outputs_of_init[2].length-1]=biases_amount;

    int nodes_already_counted_num=0;
    for (int i4=1; i4<layers.length; i4++)
    {
      nodes_counted_in_each_layer=Arrays.copyOf(nodes_counted_in_each_layer, nodes_counted_in_each_layer.length+1);
      double layers_i4=layers[i4];
      nodes_already_counted_num+=layers[i4-1];
      nodes_counted_in_each_layer[nodes_counted_in_each_layer.length-1]=layers_i4+nodes_already_counted_num;
    }

    outputs_of_init[3]=Arrays.copyOf(outputs_of_init[3], outputs_of_init[3].length+1);
    outputs_of_init[3]=nodes_counted_in_each_layer;
    return (double[][])outputs_of_init;
  }


  public static double[][][][] run_network(double[] full_population_weights, double[] full_population_biases, double[] INPUTS, int[] layers, String activation_functions, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer)
  {
    double[] running_network_weights=full_population_weights;
    double[] running_network_biases=full_population_biases;
    double[] node_firing_numbers={};
    double[][][] weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer=empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer;
    double[] NODES_PROSCESSED=INPUTS;
    int node_firing_numbers_LEN=0;
    int nodes_proscessing_LENGTH=0;
    for (int node_firing_numbers_len_finder=0; node_firing_numbers_len_finder<layers.length; node_firing_numbers_len_finder++)
    {
        node_firing_numbers_LEN+=layers[node_firing_numbers_len_finder];
        if (nodes_proscessing_LENGTH<layers[node_firing_numbers_len_finder])
        {
          nodes_proscessing_LENGTH=layers[node_firing_numbers_len_finder];
        }
    }
    node_firing_numbers=Arrays.copyOf(node_firing_numbers, node_firing_numbers_LEN);
    for (int i=0; i<layers[0]; i++)
    {
      node_firing_numbers[i]=INPUTS[i];
    }
    int computationoal_part_weights=0;
    int computationoal_part_biases=0;

    for (int i1=0; i1<layers.length-1;i1++)
    {
      double[] nodes_proscessing={};
      nodes_proscessing=Arrays.copyOf(nodes_proscessing, nodes_proscessing_LENGTH);
      for (int i2=0; i2<layers[i1+1]; i2++)
      {
        double node_in_next_layer_VALUE=0;
        for (int i3=0; i3<layers[i1]; i3++)
        {
          double edge_value=running_network_weights[computationoal_part_weights]*NODES_PROSCESSED[i3];
          weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer[i1][i2][i3]=running_network_weights[computationoal_part_weights];
          computationoal_part_weights++;
          node_in_next_layer_VALUE+=edge_value;
        }
        nodes_proscessing[i2]=node_in_next_layer_VALUE+running_network_biases[computationoal_part_biases];
        node_firing_numbers[computationoal_part_biases+layers[0]]=node_in_next_layer_VALUE+running_network_biases[computationoal_part_biases];
        computationoal_part_biases++;
      }
      NODES_PROSCESSED=nodes_proscessing;
    }
    double[] outputs=NODES_PROSCESSED;

    activation_functions=activation_functions.toLowerCase();
    if (activation_functions.contains("sigmoid"))
    {
      for (int i4=0; i4<layers[layers.length-1]; i4++)
      {
        outputs[i4]=1/(1+(Math.pow(2.71828, -1*outputs[i4])));
      }
    }
    if ((activation_functions.contains("binary_step")) || activation_functions.contains("binarystep"))
    {
      for (int i5=0; i5<layers[layers.length-1]; i5++)
      {
        if (outputs[i5]>0)
        {
          outputs[i5]=1;
        }
        else
        {
          outputs[i5]=0;
        }
      }
    }
    if (activation_functions.contains("tanh") || activation_functions.contains("hyperbolic_tangent") || activation_functions.contains("hyperbolictangent"))
    {
      for (int i6=0; i6<layers[layers.length-1]; i6++)
      {
        outputs[i6]=((Math.pow(2.71828, outputs[i6]))-(Math.pow(2.71828, -1*outputs[i6])))/((Math.pow(2.71828, outputs[i6]))+(Math.pow(2.71828, -1*outputs[i6])));
      }
    }
    if (activation_functions.contains("softplus"))
    { 
      for (int i7=0; i7<layers[layers.length-1]; i7++)
      {
        outputs[i7]=Math.log(1+(Math.pow(2.71828, outputs[i7])));
      }
    }
    if (activation_functions.contains("gaussian") || activation_functions.contains("gaussian_function") || activation_functions.contains("gaussianfunction"))
    {
      for (int i8=0; i8<layers[layers.length-1]; i8++)
      {
        outputs[i8]=Math.pow(Math.pow(2.71828, 0-outputs[i8]), 2);
      }
    }

    double[][][][] all_function_outputs={{{outputs}},{{node_firing_numbers}},weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer};
    return all_function_outputs;
  }



  public static double[] find_costs(double[] predictions, double[] goals)
  {
    double[] costs={};
    for (int i=0; i<predictions.length; i++)
    {
      costs=Arrays.copyOf(costs, costs.length+1);
      costs[costs.length-1]=Math.pow(predictions[i]-goals[i], 2);
    }

    return costs;
  }



  public static double test(double[][][] test_data, int[] layers, String activation_functions, double[] full_population_weights, double[] full_population_biases, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer)
  {
    double error=0;
    for (int i=0; i<test_data.length; i++)
    {
      for (int i1=0; i1<layers[layers.length-1]; i1++)
      {
        error+=Math.pow(run_network(full_population_weights, full_population_biases, test_data[i][0], layers, activation_functions, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer)[0][0][0][i1]-test_data[i][1][i1], 2);
      }
    }

    double average_error=error/test_data.length;
    return average_error;
  }

  public static double[][][][] train_test_data_split(double[][][] all_data, double[] ratios_of_data)
  {
    double[][][] training_data={};
    double[][][] testing_data={};
    int[] testing_data_IDs={};
    String testing_data_ID_STRING=" ";
    int appended_into_testing_data_id_string=0;
    while (appended_into_testing_data_id_string+1<all_data.length*ratios_of_data[0])
    {
      int random_ID=(int)Math.round(Math.random()*(all_data.length-1));
      if (!testing_data_ID_STRING.contains(" "+Integer.toString(random_ID)+" "))
      {
      testing_data_ID_STRING+=Integer.toString(random_ID)+" ";
      testing_data=Arrays.copyOf(testing_data, testing_data.length+1);
      testing_data[testing_data.length-1]=all_data[random_ID];
      appended_into_testing_data_id_string+=1;
      }
    }
    testing_data_ID_STRING+=" ";
    for (int i=0; i<all_data.length; i++)
    {
      if (!testing_data_ID_STRING.contains(" "+Integer.toString(i)+" "))
      {
        training_data=Arrays.copyOf(training_data, training_data.length+1);
        training_data[training_data.length-1]=all_data[i];
     }
    }

    double[][][][] function_outputs={training_data, testing_data};
    return function_outputs;
  }



  public static double[][] find_which_way_to_nudge_values_SPECIAL_FOR_GANs(int[] layers, int DATA_INSTANCE, String activation_functions, double[] full_population_weights, double[] full_population_biases, double[] empty_derivative_list_weights, double[] empty_derivative_lists_biases, double[][][] training_data, double[] last_layer_to_cost_effects_empty, double[] weight_surrounding_layer_numbers_empty, double[] nodes_counted_in_each_layer, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer, boolean generator_training, double[] full_population_weights_descriminator_ONLY_USE_IF_NEEDED, double[] full_population_biases_descriminator_ONLY_USE_IF_NEEDED, String activation_functions_descriminator_ONLY_USE_IF_NEEDED, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_descriminator_ONLY_USE_IF_NEEDED, int[] layers_descriminator_ONLY_USE_IF_NEEDED)
  {
    activation_functions=activation_functions.replace("binary_step","");
    activation_functions=activation_functions.replace("binarystep", "");
    double[][][][] run_network_packaged_outputs;
    if(!generator_training)
    {
      run_network_packaged_outputs=run_network(full_population_weights, full_population_biases, training_data[DATA_INSTANCE][0], layers, activation_functions, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer);
    }
    else
    {
      run_network_packaged_outputs=run_network(full_population_weights_descriminator_ONLY_USE_IF_NEEDED, full_population_biases_descriminator_ONLY_USE_IF_NEEDED, run_network(full_population_weights, full_population_biases, training_data[DATA_INSTANCE][0], layers, activation_functions, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer)[0][0][0], layers_descriminator_ONLY_USE_IF_NEEDED, activation_functions_descriminator_ONLY_USE_IF_NEEDED, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_descriminator_ONLY_USE_IF_NEEDED);
    }
    double[][][] weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer=run_network_packaged_outputs[2];
    double[] outputs=run_network_packaged_outputs[0][0][0];
    double[] node_firing_numbers=run_network_packaged_outputs[1][0][0];

    double[] last_layer_to_cost_effects=last_layer_to_cost_effects_empty;
    for(int i=0; i<training_data[0][1].length; i++)
    {        
      double current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=0.0; //JUST MAKING IT SO IT EXISTS FOR THE SCOPE IT IS NEEDED IN.

      if (!generator_training)
      {
        current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=2*(outputs[i]-training_data[DATA_INSTANCE][1][i]);
      }
      else
      {
        current_node_derivative_appending_to_originoal_last_layer_to_cost_effects=2*(1-outputs[i]);
      }
      if (activation_functions.contains("sigmoid"))
      {
        current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*=(1/(1+Math.pow(2.71828, -outputs[i])))*(1-(1/(1+Math.pow(2.71828, -outputs[i]))));
      }
      if ((activation_functions.contains("binary_step")) || activation_functions.contains("binarystep"))
      {
        current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*=2*((outputs[i]-training_data[DATA_INSTANCE][1][i])*100);
      }
      if (activation_functions.contains("tanh") || activation_functions.contains("hyperbolic_tangent") || activation_functions.contains("hyperbolictangent"))
      {
        current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*=1-Math.pow(((Math.pow(2.71828, outputs[i]))-(Math.pow(2.71828, -1*outputs[i])))/((Math.pow(2.71828, outputs[i]))+(Math.pow(2.71828, -1*outputs[i]))), 2);
      }
      if (activation_functions.contains("softplus"))
      { 
        current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*=1/(1+(Math.pow(2.71828, -1*outputs[i])));
      }
      if (activation_functions.contains("gaussian") || activation_functions.contains("gaussian_function") || activation_functions.contains("gaussianfunction"))
      {
        current_node_derivative_appending_to_originoal_last_layer_to_cost_effects*=-2*outputs[i]*Math.pow((Math.pow(2.71828, -outputs[i])), 2);
      }
      last_layer_to_cost_effects[i]=current_node_derivative_appending_to_originoal_last_layer_to_cost_effects;
    }
    int last_layer_to_cost_effects_LEN=layers[layers.length-1];
    int biases_went_back_counter=0;
    int weights_went_back_counter=0;
    double influence_amount=0; //making it avalible for the whole funcion... widening scope of var
    double summed_influence_on_next_layer=0; //making it avalible for the whole funcion... widening scope of var
    double effect_of_bias=0; //making it avalible for the whole funcion... widening scope of var
    double[] derivative_list_weights=empty_derivative_list_weights;
    double[] derivative_list_biases=empty_derivative_lists_biases;
    for(int i1=0; i1<layers.length-1; i1++)
    {
      for (int i2=0; i2<layers[layers.length-i1-1]; i2++)
      {
        if (i1==0)
        {
          influence_amount=-1;
        }
        else
        {
          influence_amount=layers[layers.length-i1];
        }
        summed_influence_on_next_layer=0;
        if (!(influence_amount==-1))
        {
          for(int i4=0; i4<influence_amount; i4++)
          {
            double weight_carrying_effect_value=weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer[layers.length-i1-1][i4][i2];
            double effects_in_the_last_layer=weight_carrying_effect_value*last_layer_to_cost_effects[i4];
            summed_influence_on_next_layer+=effects_in_the_last_layer;
          }
          last_layer_to_cost_effects[i2]=summed_influence_on_next_layer;
        }
        else
        {
          effect_of_bias=last_layer_to_cost_effects[i2];
          last_layer_to_cost_effects[i1]=effect_of_bias;
        }
      }
            
      if (!(influence_amount==-1))
      {
      last_layer_to_cost_effects_LEN=layers[layers.length-i1-1];
      }
      for(int i3=0; i3<(last_layer_to_cost_effects.length-last_layer_to_cost_effects_LEN); i3++)
      {
        last_layer_to_cost_effects[i3+last_layer_to_cost_effects_LEN]=(double)0;
      }

      for (int i6=0; i6<layers[layers.length-i1-1]; i6++)
      {
        for(int i7=0; i7<layers[layers.length-i1-2]; i7++)
        {
          derivative_list_weights[derivative_list_weights.length-weights_went_back_counter-1]=(last_layer_to_cost_effects[last_layer_to_cost_effects_LEN-i6-1])*node_firing_numbers[(int)((layers[layers.length-i1-2]-i7-1)+nodes_counted_in_each_layer[layers.length-i1-2])];
          weights_went_back_counter++;
        }
      }
      for (int i5=0; i5<layers[layers.length-1-i1]; i5++)
      {
        derivative_list_biases[derivative_list_biases.length-biases_went_back_counter-1]=last_layer_to_cost_effects[i5];
        biases_went_back_counter++;
      }


    }
    double[][] packaged_outputs={{},{}};
    packaged_outputs[1]=derivative_list_biases;
    packaged_outputs[0]=derivative_list_weights;

    return packaged_outputs;
  }




  public static double[][] take_gradient_decent_step_SPECIAL_FOR_GANs(int[] layers, String activation_functions, double[][][] training_data, double weights_amount, double biases_amount, double[] full_population_weights, double[] full_population_biases, double learning_rate, double[] empty_derivative_list_weights, double[] empty_derivative_lists_biases, double[] last_layer_to_cost_effects_empty, double[] weight_surrounding_layer_numbers_empty, double[] nodes_counted_in_each_layer,double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer, boolean generator_training, double[] full_population_weights_descriminator_ONLY_USE_IF_NEEDED, double[] full_population_biases_descriminator_ONLY_USE_IF_NEEDED, String activation_functions_descriminator_ONLY_USE_IF_NEEDED, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_descriminator_ONLY_USE_IF_NEEDED,  int[] layers_descriminator_ONLY_USE_IF_NEEDED)
  {
    double[][] packaged_outputs={full_population_weights,full_population_biases};

    for(int i=0; i<training_data.length; i++)
    {
      double[][] weights_and_biases_packaged_derivatives=find_which_way_to_nudge_values_SPECIAL_FOR_GANs(layers, i, activation_functions, full_population_weights, full_population_biases, empty_derivative_list_weights, empty_derivative_lists_biases, training_data, last_layer_to_cost_effects_empty, weight_surrounding_layer_numbers_empty, nodes_counted_in_each_layer, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer, generator_training, full_population_weights_descriminator_ONLY_USE_IF_NEEDED, full_population_biases_descriminator_ONLY_USE_IF_NEEDED, activation_functions_descriminator_ONLY_USE_IF_NEEDED, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_descriminator_ONLY_USE_IF_NEEDED, layers_descriminator_ONLY_USE_IF_NEEDED);
      double[] derivatives_in_network_weights=weights_and_biases_packaged_derivatives[0];
      double[] derivatives_in_network_biases=weights_and_biases_packaged_derivatives[1];
      for (int i1=0; i1<weights_amount; i1++)
      {
        packaged_outputs[0][i1]+=derivatives_in_network_weights[i1]*learning_rate*-1/training_data.length;
      }
      for(int i2=0; i2<biases_amount; i2++)
      {
        packaged_outputs[1][i2]+=(derivatives_in_network_biases[i2]*learning_rate*-1)/training_data.length;
      }
    }



    return packaged_outputs;
  }



  public static double[][] empty_deriavative_lists_for_speed(double weights_amount, double biases_amount)
  {
    double[] empty_weights_derivative_list={};
    double[] empty_biases_derivative_list={};
    for (int i=0; i<weights_amount; i++)
    {
      empty_weights_derivative_list=Arrays.copyOf(empty_weights_derivative_list, empty_weights_derivative_list.length+1);
      empty_weights_derivative_list[empty_weights_derivative_list.length-1]=(double) 0;
    }
    for (int i1=0; i1<biases_amount; i1++)
    {
      empty_biases_derivative_list=Arrays.copyOf(empty_biases_derivative_list, empty_biases_derivative_list.length+1);
      empty_biases_derivative_list[empty_biases_derivative_list.length-1]=(double) 0;
    }
    double[][] function_outupts={empty_weights_derivative_list, empty_biases_derivative_list};
    return function_outupts;
  }

  public static double[][] last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency(int[] layers_for_efficiency)
  {
    double[] last_layer_to_cost_effects_EMPTY={};
    double[] weight_surrounding_layer_numbers_EMPTY={0,0};
    int max_layer_sums_ID_of_first_layer=-1;
    int max_sum=0;
    for(int i1=0; i1<layers_for_efficiency.length-1; i1++)
    {
      if((layers_for_efficiency[i1]+layers_for_efficiency[i1+1])>max_sum)
      {
        max_layer_sums_ID_of_first_layer=i1;
        max_sum=(layers_for_efficiency[i1]+layers_for_efficiency[i1+1]);
      }
    }
    for(int i=0; i<max_sum; i++)
    {
      last_layer_to_cost_effects_EMPTY=Arrays.copyOf(last_layer_to_cost_effects_EMPTY, last_layer_to_cost_effects_EMPTY.length+1);
      last_layer_to_cost_effects_EMPTY[last_layer_to_cost_effects_EMPTY.length-1]=(double)0;
    }
    int[] layers=layers_for_efficiency;
    double[][] function_outputs={last_layer_to_cost_effects_EMPTY, weight_surrounding_layer_numbers_EMPTY};
    return function_outputs;
  }

  public static String get_text_data()
  {
      BufferedReader file_contents;
      String file_contents_string="";
      try
      {
          file_contents=new BufferedReader(new FileReader(("C:\\Users\\andre\\OneDrive\\Desktop\\text_to_array_file_converter_file\\GAN_training_data.txt"))); //DO NOT CHANGE THIS PATH!!!!!
          file_contents_string=file_contents.readLine();
      }
      catch (Exception ex)
      {

      }
      return file_contents_string;
  }

  public static double[][][] convert_data(String text_file_text_in_string, int data_instance_amount, int network_inputs_amount,  int network_outputs_amount, boolean print_data)
  {
      //getting ready for main conversion loop
      text_file_text_in_string=text_file_text_in_string.replace(" ", "");
      double[][][] end_array={};
      double[][] sample_data_instance_empty={{},{}};
      end_array=Arrays.copyOf(end_array, data_instance_amount);
      for (int end_array_making_for_loop=0; end_array_making_for_loop<data_instance_amount; end_array_making_for_loop++)
      {
          end_array[end_array_making_for_loop]=sample_data_instance_empty;
      }
      int character_counter_passed_all=1;


      //main conversion loop
      for(int i=0; i<data_instance_amount; i++)
      {
          boolean end_of_instance_reached=false; //FOR BOTH INPUTS AND OUTPUTS
          int character_counter_passed_data_intsance=0; //FOR BOTH INPUTS AND OUTPUTS
          int array_ends_found_so_far=0; //FOR BOTH INPUTS AND OUTPUTS

          String string_version_of_what_to_convert_INPUTS=""; //FOR INPUTS
          int inputs_finished_so_far=0; //FOR ONLY INPUTS
          boolean currently_REALLY_inside_inputs=false; //FOR ONLY INPUTS
          boolean currently_counting_inside_inputs=false; //FOR ONLY INPUTS
          double[] current_inputs_finding={}; //FOR ONLY INPUTS
          current_inputs_finding=Arrays.copyOf(current_inputs_finding, network_inputs_amount); //FOR ONLY INPUTS

          String string_version_of_what_to_convert_OUTPUTS=""; //FOR OUTPUTS
          boolean currently_inside_outputs=false; //FOR ONLY OUTPUTS
          boolean currently_REALLY_inside_outputs=false; //FOR ONLY OUTPUTS
          int outputs_finished_so_far=0; //FOR ONLY OUTPUTS
          double[] current_outputs_finding={};
          current_outputs_finding=Arrays.copyOf(current_outputs_finding, network_outputs_amount);

          while(end_of_instance_reached==false)
          {
              if((text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)==',') && (currently_counting_inside_inputs==true)) //FOR INPUTS. CHANGE TO MAKE IT DIFFERENT FROM OUTPTUS!!!
              {
                  currently_counting_inside_inputs=false;
                  double double_converted_inputs=Double.parseDouble(string_version_of_what_to_convert_INPUTS);
                  current_inputs_finding[inputs_finished_so_far]=double_converted_inputs;
                  inputs_finished_so_far+=1;
                  string_version_of_what_to_convert_INPUTS="";
              }
              if ((text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)=='}') && (currently_REALLY_inside_inputs=true)) //FOR INPUTS
              {
                  double double_converted_inputs;
                  currently_counting_inside_inputs=false;
                  currently_REALLY_inside_inputs=false;
                  try
                  {
                      double_converted_inputs=Double.parseDouble(string_version_of_what_to_convert_INPUTS);
                      current_inputs_finding[inputs_finished_so_far]=double_converted_inputs;
                  }
                  catch (Exception ex)
                  {
                      double_converted_inputs=0.0;
                  }
              }

              if((text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)==',') && (currently_inside_outputs==true)) //FOR OUTPUTS.
              {
                  currently_inside_outputs=false;
                  double double_converted_outputs=Double.parseDouble(string_version_of_what_to_convert_OUTPUTS);
                  current_outputs_finding[outputs_finished_so_far]=double_converted_outputs;
                  outputs_finished_so_far+=1;
                  string_version_of_what_to_convert_OUTPUTS="";
              }
              if ((text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)=='}') && (currently_REALLY_inside_outputs=true)) //FOR OUTPUTS
              {
                  double double_converted_outputs;
                  currently_inside_outputs=false;
                  currently_REALLY_inside_outputs=false;
                  try
                  {
                      double_converted_outputs=Double.parseDouble(string_version_of_what_to_convert_OUTPUTS);
                      current_outputs_finding[outputs_finished_so_far]=double_converted_outputs;
                  }
                  catch (Exception ex)
                  {
                      double_converted_outputs=0.0;
                  }
              }

              
              if (text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)=='}') //FOR BOTH INPUTS AND OUTPUTS
              {
                  array_ends_found_so_far+=1;
                  currently_counting_inside_inputs=false;
                  currently_REALLY_inside_inputs=false;
                  currently_inside_outputs=false;
                  currently_REALLY_inside_outputs=false;
                  string_version_of_what_to_convert_INPUTS="";
                  string_version_of_what_to_convert_OUTPUTS="";
              }
              if(array_ends_found_so_far==3) //FOR BOTH INPUTS AND OUTPUTS
              {
                  end_of_instance_reached=true;
              }

              if (currently_counting_inside_inputs==true) //FOR ONLY INPUTS
              {
                  string_version_of_what_to_convert_INPUTS=string_version_of_what_to_convert_INPUTS.concat(String.valueOf(text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)));
              }
              if (currently_inside_outputs==true) //FOR ONLY OUTUPTS
              {
                  string_version_of_what_to_convert_OUTPUTS=string_version_of_what_to_convert_OUTPUTS.concat(String.valueOf(text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)));
              }


              if((text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)=='{') && (text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance-1)=='{') && !(character_counter_passed_all+character_counter_passed_data_intsance-2==-1)) //for inputs
              {
                  currently_counting_inside_inputs=true;
                  currently_REALLY_inside_inputs=true;
                  string_version_of_what_to_convert_INPUTS="";
              }
              if ((text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)=='{') && (text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance-1)==',') && !(text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance-3)=='}')) //for outputs
              {   
                  currently_inside_outputs=true;
                  currently_REALLY_inside_outputs=true;
                  string_version_of_what_to_convert_OUTPUTS="";
              }

              if ((text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)==',') && currently_REALLY_inside_inputs) //FOR INPUTS
              {
                  currently_counting_inside_inputs=true;
              }
              if ((text_file_text_in_string.charAt(character_counter_passed_all+character_counter_passed_data_intsance)==',') && currently_REALLY_inside_outputs) //FOR OUTUPTS
              {
                  currently_inside_outputs=true;
              }
              character_counter_passed_data_intsance+=1;
          }
          end_array[i]=new double[][] {current_inputs_finding, current_outputs_finding};
          character_counter_passed_all+=character_counter_passed_data_intsance;
      }

      //printing stuff and returning
      if (print_data==true)
      {
      System.out.print("Text file data: ");
      System.out.println(text_file_text_in_string);
      }
      return end_array;
  }

  public static double[][][] initialize_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed(int[] layers)
  {
    double[][][] output_array={};
    double[][] new_weights_and_biases_discriminator=null;
    double[] new_weights_discriminator=null;
    double[] new_biases_discriminator=null;
    output_array=Arrays.copyOf(output_array, layers.length-1);
    for (int i=0; i<output_array.length; i++)
    {
      output_array[i]=Arrays.copyOf(new double[][] {}, layers[i+1]);
      for (int i1=0; i1<output_array[i].length; i1++)
      {
        output_array[i][i1]=new double[] {};
        output_array[i][i1]=Arrays.copyOf(output_array[i][i1], layers[i]);
      }
    }
    
    return output_array;
  }


  public static double[][] romance_emotion(double[] full_population_weights_discriminator, double[] full_population_biases_discriminator, double[][][] outputs_for_all_training_data_generator_FAKE, int[] LAYERS_BEING_USED_DISCRIMINATOR, String activation_functions_for_discriminator, double[][][] initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator, double[][][] real_data, int extra_iterations_equalizing_romance, double learning_rate_descriminator, double[] empty_derivative_list_weights_dicsriminator, double[] empty_derivative_lists_biases_discriminator, double[] last_layer_to_cost_effects_empty_discriminator, double[] weight_surrounding_layer_numbers_empty_discriminator, double[] nodes_counted_in_each_layer_discriminator, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_discriminator,             double[] full_population_weights_generator, double[] full_population_biases_generator, int[] LAYERS_BEING_USED_GENERATOR, String activation_functions_for_generator, double[][][] initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator, double learning_rate_generator, double[] empty_derivative_list_weights_generator, double[] empty_derivative_lists_biases_generator, double[] last_layer_to_cost_effects_empty_generator, double[] weight_surrounding_layer_numbers_empty_generator, double[] nodes_counted_in_each_layer_generator, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_generator, double weights_amount_generator, double biases_amount_generator)
  {
    double overall_fake_tricking_score=0;
    boolean generator_ahead=true;
    double[] new_weights_discriminator=full_population_weights_discriminator;
    double[] new_biases_discriminator=full_population_biases_discriminator;
    double[] new_weights_generator=full_population_weights_generator;
    double[] new_biases_generator=full_population_biases_generator;
    for (int i=0; i<outputs_for_all_training_data_generator_FAKE.length; i++)
    {
      overall_fake_tricking_score+=run_network(full_population_weights_discriminator, full_population_biases_discriminator, outputs_for_all_training_data_generator_FAKE[i][0], LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator)[0][0][0][0];
    }
    double overall_real_tricking_score=0;
    for (int i1=0; i1<real_data.length; i1++)
    {
      overall_real_tricking_score+=run_network(full_population_weights_discriminator, full_population_biases_discriminator, real_data[i1][0], LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator)[0][0][0][0];
    }

    if (overall_fake_tricking_score<overall_real_tricking_score)
    {
      generator_ahead=false;
      System.out.print("The descriminator is ahead");
        }   
    else
    {
      if (overall_fake_tricking_score>overall_real_tricking_score)
      {
        generator_ahead=true;
        System.out.print("The generator is ahead");
      }
    }

    if (generator_ahead==true)
    {
      int data_instance_amount_generator=outputs_for_all_training_data_generator_FAKE.length;
      int data_instance_amount_real=real_data.length;
      double[][][] discriminator_training_data={}; //This is the real end result one
      double[][][] discriminator_training_data_DRAFT={}; //This one is sorted by only false then only true, but it makes it so the real data can be scrambled in terms of real and fake.
      discriminator_training_data=Arrays.copyOf(new double[][][] {}, data_instance_amount_generator+data_instance_amount_real);
      discriminator_training_data_DRAFT=Arrays.copyOf(new double[][][] {}, data_instance_amount_generator+data_instance_amount_real);
      for (int i3=0; i3<data_instance_amount_real; i3++)
      {
        discriminator_training_data_DRAFT[i3]=Arrays.copyOf(new double[][] {{},{}}, data_instance_amount_generator);
        discriminator_training_data_DRAFT[i3][0]=outputs_for_all_training_data_generator_FAKE[i3][0];
        discriminator_training_data_DRAFT[i3][1]=outputs_for_all_training_data_generator_FAKE[i3][1];
      }
      for (int i4=0; i4<data_instance_amount_real; i4++)
      {
        discriminator_training_data_DRAFT[i4+data_instance_amount_generator]=Arrays.copyOf(new double[][] {{},{}}, data_instance_amount_real);
        discriminator_training_data_DRAFT[i4+data_instance_amount_generator][0]=real_data[i4][0];
        discriminator_training_data_DRAFT[i4+data_instance_amount_generator][1]=new double[] {1};
      }
      for (int i2=0; i2<data_instance_amount_generator+data_instance_amount_real; i2++)
      {
        String data_ID_STRING=" ";
        for (int i5=0; i5<data_instance_amount_generator+data_instance_amount_real; i5++)
        {
          int random_ID=(int)Math.round(Math.random()*(discriminator_training_data_DRAFT.length-1));
          if (!data_ID_STRING.contains(" "+Integer.toString(random_ID)+" "))
          {
          data_ID_STRING+=Integer.toString(random_ID)+" ";
          discriminator_training_data[i2]=discriminator_training_data_DRAFT[random_ID];
          }
        }
      }

      for (int i2=0; i2<extra_iterations_equalizing_romance; i2++)
      { 
        
        double[][] new_weights_and_biases_discriminator=take_gradient_decent_step_SPECIAL_FOR_GANs(LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, discriminator_training_data, full_population_weights_discriminator.length, full_population_biases_discriminator.length, full_population_weights_discriminator, full_population_biases_discriminator, learning_rate_descriminator,  empty_derivative_list_weights_dicsriminator, empty_derivative_lists_biases_discriminator, last_layer_to_cost_effects_empty_discriminator, weight_surrounding_layer_numbers_empty_discriminator, nodes_counted_in_each_layer_discriminator, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_discriminator, false, new double[] {}, new double[] {}, new String (""), new double[][][] {}, new int[] {});
        new_weights_discriminator=new_weights_and_biases_discriminator[0];
        new_biases_discriminator=new_weights_and_biases_discriminator[1];
        full_population_weights_discriminator=new_weights_discriminator;
        full_population_biases_discriminator=new_biases_discriminator;
      }
    } 
    else
    {
      for (int i3=0; i3<extra_iterations_equalizing_romance; i3++)
      {
        double[][] new_weights_and_biases_generator=take_gradient_decent_step_SPECIAL_FOR_GANs(LAYERS_BEING_USED_GENERATOR, activation_functions_for_generator, outputs_for_all_training_data_generator_FAKE, full_population_weights_generator.length, full_population_biases_generator.length, full_population_weights_generator, full_population_biases_generator, learning_rate_generator, empty_derivative_list_weights_generator, empty_derivative_lists_biases_generator, last_layer_to_cost_effects_empty_generator, weight_surrounding_layer_numbers_empty_generator, nodes_counted_in_each_layer_generator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator, true, full_population_weights_discriminator, full_population_biases_discriminator, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator, LAYERS_BEING_USED_DISCRIMINATOR);
        new_weights_generator=new_weights_and_biases_generator[0];
        new_biases_generator=new_weights_and_biases_generator[1];
        full_population_weights_generator=new_weights_generator;
        full_population_biases_generator=new_biases_generator;
      }
    }


    double[][] romance_outputs=new double[][] {new_weights_discriminator, new_biases_discriminator, new_weights_generator, new_biases_generator};
    return romance_outputs;
  }


  public static double[][][] control_emotions(double[] full_population_weights_discriminator, double[] full_population_biases_discriminator, double[][][] outputs_for_all_training_data_generator_FAKE, int[] LAYERS_BEING_USED_DISCRIMINATOR, String activation_functions_for_discriminator, double[][][] initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator, double[][][] real_data, int extra_iterations_equalizing_romance, double learning_rate_descriminator, double[] empty_derivative_list_weights_dicsriminator, double[] empty_derivative_lists_biases_discriminator, double[] last_layer_to_cost_effects_EMPTY_discriminator, double[] weight_surrounding_layer_numbers_EMPTY_discriminator, double[] nodes_counted_in_each_layer_discriminator, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_discriminator, double[] full_population_weights_generator, double[] full_population_biases_generator, int[] LAYERS_BEING_USED_GENERATOR, String activation_functions_for_generator, double[][][] initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator, double learning_rate_generator, double[] empty_derivative_list_weights_generator, double[] empty_derivative_lists_biases_generator, double[] last_layer_to_cost_effects_empty_generator, double[] weight_surrounding_layer_numbers_empty_generator, double[] nodes_counted_in_each_layer_generator, double[][][] empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_generator, double weights_amount_generator, double biases_amount_generator)
  {
    double[][] romance_outputs=romance_emotion(full_population_weights_discriminator, full_population_biases_discriminator, outputs_for_all_training_data_generator_FAKE, LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator, real_data, extra_iterations_equalizing_romance, learning_rate_descriminator, empty_derivative_list_weights_dicsriminator, empty_derivative_lists_biases_discriminator, last_layer_to_cost_effects_EMPTY_discriminator, weight_surrounding_layer_numbers_EMPTY_discriminator, nodes_counted_in_each_layer_discriminator, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_discriminator, full_population_weights_generator, full_population_biases_generator, LAYERS_BEING_USED_GENERATOR, activation_functions_for_generator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator, learning_rate_generator, empty_derivative_list_weights_generator, empty_derivative_lists_biases_generator, last_layer_to_cost_effects_empty_generator, weight_surrounding_layer_numbers_empty_generator, nodes_counted_in_each_layer_generator, empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_generator, weights_amount_generator, biases_amount_generator);
    return new double[][][] {romance_outputs};
  }


    public static void main(String[] args)
    {
        //things you can specify
        System.out.println("Retreiving data...");
        double[] initializing_range_discriminator={-0.3,0.3};
        double[] initializing_range_generator={-0.3,0.3};
        String data_as_string=get_text_data();
        //THE AMOUNT OF INPUTS IN THE DISCRIMINATOR SHOULD BE THE SAME AS THE AMOUNT OF OUTPUTS IN THE GENERATOR, AND THE AMOUNT OF OUTPUTS IN THE DESCRIMINATOR SHOULD BE 1.
        int[] LAYERS_BEING_USED_DISCRIMINATOR={1,5,5,1};
        int[] LAYERS_BEING_USED_DISCRIMINATOR_FOR_EFFICIENCY={1,5,5,1};
        int[] LAYERS_BEING_USED_GENERATOR={1,2,5,1};
        int[] LAYERS_BEING_USED_GENERATOR_FOR_EFFICIENCY={1,2,5,1};
        String activation_functions_for_generator="";
        String additional_activation_functions_discriminator="";
        //the two data instance amounts
        int data_instance_amount_real=2;
        int data_instance_amount_generator=2;
        boolean print_data=true;
        double[][][] data=convert_data(data_as_string, data_instance_amount_real, LAYERS_BEING_USED_DISCRIMINATOR[0], LAYERS_BEING_USED_DISCRIMINATOR[LAYERS_BEING_USED_DISCRIMINATOR.length-1], print_data);
        double[] testing_training_data_ratio={0,1};
        int extra_iterations_equalizing_romance=1;
        final int epoch_amount=1000000000; //1000000000 is the max factor of 10 that can be in this, because otherwise it would be a long or some other type of datatype, but this is for all practical uses infinity.
        final double learning_rate_discriminator=0.00001;
        final double learning_rate_generator=0.00001;

        //Getting ready for main computation
        System.out.println("Getting ready...");
        String activation_functions_for_discriminator="sigmoid "+additional_activation_functions_discriminator;
        double[][] all_outputs_of_init_discriminator=init(LAYERS_BEING_USED_DISCRIMINATOR, initializing_range_discriminator);
        double[][] all_outputs_of_init_generator=init(LAYERS_BEING_USED_GENERATOR, initializing_range_generator);
        double[][][] initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator=initialize_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed(LAYERS_BEING_USED_DISCRIMINATOR);
        double[][][] initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator=initialize_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed(LAYERS_BEING_USED_GENERATOR);
        double[][][][] train_test_data_split_outputs=train_test_data_split(data, testing_training_data_ratio);
        double[][][] training_data=train_test_data_split_outputs[0];
        double[][][] testing_data=train_test_data_split_outputs[1];
        double[] full_population_weights_discriminator=all_outputs_of_init_discriminator[0];
        double[] full_population_biases_discriminator=all_outputs_of_init_discriminator[1];
        double[] full_population_weights_generator=all_outputs_of_init_generator[0];
        double[] full_population_biases_generator=all_outputs_of_init_generator[1];
        double weights_amount_discriminator=all_outputs_of_init_discriminator[2][0];
        double biases_amount_discriminator=all_outputs_of_init_discriminator[2][1];
        double weights_amount_generator=all_outputs_of_init_generator[2][0];
        double biases_amount_generator=all_outputs_of_init_generator[2][1];
        double[] nodes_counted_in_each_layer_discriminator=all_outputs_of_init_discriminator[3];
        double[] nodes_counted_in_each_layer_generator=all_outputs_of_init_generator[3];
        double[][] empty_derivative_lists_discriminator=empty_deriavative_lists_for_speed(weights_amount_discriminator, biases_amount_discriminator);
        double[][] empty_derivative_lists_generator=empty_deriavative_lists_for_speed(weights_amount_generator, biases_amount_generator);
        double[] empty_derivative_list_weights_discriminator=empty_derivative_lists_discriminator[0];
        double[] empty_derivative_lists_biases_discriminator=empty_derivative_lists_discriminator[1];
        double[] empty_derivative_list_weights_generator=empty_derivative_lists_generator[0];
        double[] empty_derivative_lists_biases_generator=empty_derivative_lists_generator[1];
        double[][] packaged_last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency_outputs_discriminator=last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency(LAYERS_BEING_USED_DISCRIMINATOR_FOR_EFFICIENCY);
        double[][] packaged_last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency_outputs_generator=last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency(LAYERS_BEING_USED_GENERATOR_FOR_EFFICIENCY);
        double[] last_layer_to_cost_effects_EMPTY_discriminator=packaged_last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency_outputs_discriminator[0];
        double[] last_layer_to_cost_effects_EMPTY_generator=packaged_last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency_outputs_generator[0];
        double[] weight_surrounding_layer_numbers_EMPTY_discriminator=packaged_last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency_outputs_discriminator[1];
        double[] weight_surrounding_layer_numbers_EMPTY_generator=packaged_last_layer_to_cost_effects_and_weight_surrounding_layer_numbers_empty_lists_for_efficiency_outputs_generator[1];
        //Main training loop
        for (int i=0; i<epoch_amount; i++)
        {
          //double[][] new_weights_and_biases_discriminator=take_gradient_decent_step(LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, training_data, weights_amount_discriminator, biases_amount_discriminator, full_population_weights_discriminator, full_population_biases_discriminator, learning_rate, empty_derivative_list_weights_discriminator, empty_derivative_lists_biases_discriminator, last_layer_to_cost_effects_EMPTY_discriminator, weight_surrounding_layer_numbers_EMPTY_discriminator, nodes_counted_in_each_layer_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator);
          //full_population_weights_discriminator=new_weights_and_biases_discriminator[0];
          //full_population_biases_discriminator=new_weights_and_biases_discriminator[1];
          //System.out.println(Arrays.toString(run_network(full_population_weights_discriminator, full_population_biases_discriminator, new double[] {0.5}, LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator)[0][0][0]));
          //double[][] new_weights_and_biases_generator=take_gradient_decent_step(LAYERS_BEING_USED_GENERATOR, activation_functions_for_generator, training_data, weights_amount_generator, biases_amount_generator, full_population_weights_generator, full_population_biases_generator, learning_rate, empty_derivative_list_weights_generator, empty_derivative_lists_biases_generator, last_layer_to_cost_effects_EMPTY_generator, weight_surrounding_layer_numbers_EMPTY_generator, nodes_counted_in_each_layer_generator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator);
          ///full_population_weights_generator=new_weights_and_biases_generator[0];
          //full_population_biases_generator=new_weights_and_biases_generator[1];
          //System.out.println(Arrays.toString(run_network(full_population_weights_generator, full_population_biases_generator, new double[] {0.5}, LAYERS_BEING_USED_GENERATOR, activation_functions_for_generator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator)[0][0][0]));

          double[][][] outputs_for_all_training_data_generator_FAKE={};
          outputs_for_all_training_data_generator_FAKE=Arrays.copyOf(new double[][][] {}, data_instance_amount_generator);
          for (int i1=0; i1<data_instance_amount_generator; i1++)
          {
            outputs_for_all_training_data_generator_FAKE[i1]=Arrays.copyOf(new double[][] {{},{}}, data_instance_amount_generator);
            outputs_for_all_training_data_generator_FAKE[i1][0]=run_network(full_population_weights_generator, full_population_biases_generator, new double[] {Math.random()}, LAYERS_BEING_USED_GENERATOR, activation_functions_for_generator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator)[0][0][0];
            outputs_for_all_training_data_generator_FAKE[i1][1]=new double[] {0};
          }

          double[][][] discriminator_training_data={}; //This is the real end result one
          double[][][] discriminator_training_data_DRAFT={}; //This one is sorted by only false then only true, but it makes it so the real data can be scrambled in terms of real and fake.
          discriminator_training_data=Arrays.copyOf(new double[][][] {}, data_instance_amount_generator+data_instance_amount_real);
          discriminator_training_data_DRAFT=Arrays.copyOf(new double[][][] {}, data_instance_amount_generator+data_instance_amount_real);
          for (int i3=0; i3<data_instance_amount_real; i3++)
          {
            discriminator_training_data_DRAFT[i3]=Arrays.copyOf(new double[][] {{},{}}, data_instance_amount_generator);
            discriminator_training_data_DRAFT[i3][0]=outputs_for_all_training_data_generator_FAKE[i3][0];
            discriminator_training_data_DRAFT[i3][1]=outputs_for_all_training_data_generator_FAKE[i3][1];
          }
          for (int i4=0; i4<data_instance_amount_real; i4++)
          {
            discriminator_training_data_DRAFT[i4+data_instance_amount_generator]=Arrays.copyOf(new double[][] {{},{}}, data_instance_amount_real);
            discriminator_training_data_DRAFT[i4+data_instance_amount_generator][0]=data[i4][0];
            discriminator_training_data_DRAFT[i4+data_instance_amount_generator][1]=new double[] {1};
          }
          for (int i2=0; i2<data_instance_amount_generator+data_instance_amount_real; i2++)
          {
            String data_ID_STRING=" ";
            for (int i5=0; i5<data_instance_amount_generator+data_instance_amount_real; i5++)
            {
              int random_ID=(int)Math.round(Math.random()*(discriminator_training_data_DRAFT.length-1));
              if (!data_ID_STRING.contains(" "+Integer.toString(random_ID)+" "))
              {
              data_ID_STRING+=Integer.toString(random_ID)+" ";
              discriminator_training_data[i2]=discriminator_training_data_DRAFT[random_ID];
              }
            }
          }


          double[][] new_weights_and_biases_discriminator=take_gradient_decent_step_SPECIAL_FOR_GANs(LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, discriminator_training_data, weights_amount_discriminator, biases_amount_discriminator, full_population_weights_discriminator, full_population_biases_discriminator, learning_rate_discriminator, empty_derivative_list_weights_discriminator, empty_derivative_lists_biases_discriminator, last_layer_to_cost_effects_EMPTY_discriminator, weight_surrounding_layer_numbers_EMPTY_discriminator, nodes_counted_in_each_layer_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator, false , new double[] {}, new double[] {}, new String (""), new double[][][] {}, new int[] {}); //The false near the end indicates that it is not the generator that it is finding the error for when calculating derivatives
          full_population_weights_discriminator=new_weights_and_biases_discriminator[0];
          full_population_biases_discriminator=new_weights_and_biases_discriminator[1];
          

          System.out.print("Discriminator guesses (for fake then real data): "+Arrays.toString(run_network(full_population_weights_discriminator, full_population_biases_discriminator, outputs_for_all_training_data_generator_FAKE[0][0], LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator)[0][0][0]));
          System.out.print(Arrays.toString(run_network(full_population_weights_discriminator, full_population_biases_discriminator, data[0][0], LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator)[0][0][0]));

          
          double[][] new_weights_and_biases_generator=take_gradient_decent_step_SPECIAL_FOR_GANs(LAYERS_BEING_USED_GENERATOR, activation_functions_for_generator, data, weights_amount_generator, biases_amount_generator, full_population_weights_generator, full_population_biases_generator, learning_rate_generator, empty_derivative_list_weights_generator, empty_derivative_lists_biases_generator, last_layer_to_cost_effects_EMPTY_generator, weight_surrounding_layer_numbers_EMPTY_generator, nodes_counted_in_each_layer_generator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator, true, full_population_weights_discriminator, full_population_biases_discriminator, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator, LAYERS_BEING_USED_DISCRIMINATOR);
          full_population_weights_generator=new_weights_and_biases_generator[0];
          full_population_biases_generator=new_weights_and_biases_generator[1];
      

          double[][][] control_emotions_outputs=control_emotions(full_population_weights_discriminator, full_population_biases_discriminator, outputs_for_all_training_data_generator_FAKE, LAYERS_BEING_USED_DISCRIMINATOR, activation_functions_for_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator, data, extra_iterations_equalizing_romance, learning_rate_discriminator, empty_derivative_list_weights_discriminator, empty_derivative_lists_biases_discriminator, last_layer_to_cost_effects_EMPTY_discriminator, weight_surrounding_layer_numbers_EMPTY_discriminator, nodes_counted_in_each_layer_discriminator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_discriminator, full_population_weights_generator, full_population_biases_generator, LAYERS_BEING_USED_GENERATOR, activation_functions_for_generator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator, learning_rate_generator, empty_derivative_list_weights_generator, empty_derivative_lists_biases_generator, last_layer_to_cost_effects_EMPTY_generator, weight_surrounding_layer_numbers_EMPTY_generator, nodes_counted_in_each_layer_generator, initialized_empty_weights_sorted_in_layers_then_second_connection_in_layer_then_first_connection_in_layer_for_speed_generator, weights_amount_generator, biases_amount_generator);
          full_population_weights_discriminator=control_emotions_outputs[0][0];
          full_population_biases_discriminator=control_emotions_outputs[0][1];
          full_population_weights_generator=control_emotions_outputs[0][2];
          full_population_biases_generator=control_emotions_outputs[0][3];
          System.out.println();


         //THE TRAINING OF THE DESCRIMINATOR AND GENERATOR WORKS!!!
         //Romance works now, the romance emotion works, so it is much better and more stable.
        
        }


    }

    //remember to go to https://github.com/Algorithmic-TITAN/java_generative_adverserial_network_code to save to github (this is in the src folder in file explorer)
}
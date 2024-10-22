��	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.22unknown8��
�
1online_cnn_fw/depthwise_conv2d_6/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31online_cnn_fw/depthwise_conv2d_6/depthwise_kernel
�
Eonline_cnn_fw/depthwise_conv2d_6/depthwise_kernel/Read/ReadVariableOpReadVariableOp1online_cnn_fw/depthwise_conv2d_6/depthwise_kernel*&
_output_shapes
:*
dtype0
�
online_cnn_fw/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameonline_cnn_fw/conv2d_6/kernel
�
1online_cnn_fw/conv2d_6/kernel/Read/ReadVariableOpReadVariableOponline_cnn_fw/conv2d_6/kernel*&
_output_shapes
: *
dtype0
�
1online_cnn_fw/depthwise_conv2d_7/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31online_cnn_fw/depthwise_conv2d_7/depthwise_kernel
�
Eonline_cnn_fw/depthwise_conv2d_7/depthwise_kernel/Read/ReadVariableOpReadVariableOp1online_cnn_fw/depthwise_conv2d_7/depthwise_kernel*&
_output_shapes
: *
dtype0
�
online_cnn_fw/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_nameonline_cnn_fw/conv2d_7/kernel
�
1online_cnn_fw/conv2d_7/kernel/Read/ReadVariableOpReadVariableOponline_cnn_fw/conv2d_7/kernel*&
_output_shapes
: @*
dtype0
�
)online_cnn_fw/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)online_cnn_fw/batch_normalization_3/gamma
�
=online_cnn_fw/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp)online_cnn_fw/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
�
(online_cnn_fw/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(online_cnn_fw/batch_normalization_3/beta
�
<online_cnn_fw/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp(online_cnn_fw/batch_normalization_3/beta*
_output_shapes
:@*
dtype0
�
/online_cnn_fw/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/online_cnn_fw/batch_normalization_3/moving_mean
�
Conline_cnn_fw/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp/online_cnn_fw/batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
�
3online_cnn_fw/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53online_cnn_fw/batch_normalization_3/moving_variance
�
Gonline_cnn_fw/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp3online_cnn_fw/batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0

NoOpNoOp
�7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�7
value�7B�7 B�7
�
Convolution2D_DepthWise_0
Convolution2D_0

ReLU_0
	Dropout_0
Convolution2D_DepthWise_1
Convolution2D_1

ReLU_1
GlobalAveragePool2D_GAP
	Flatten_GAP

BatchNormalization1D_GAP
Dropout_GAP
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�
depthwise_kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,_random_generator
-__call__
*.&call_and_return_all_conditional_losses* 
�
/depthwise_kernel
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*
�

6kernel
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 

I	keras_api* 
�
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y_random_generator
Z__call__
*[&call_and_return_all_conditional_losses* 
<
0
1
/2
63
K4
L5
M6
N7*
.
0
1
/2
63
K4
L5*
* 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

aserving_default* 
��
VARIABLE_VALUE1online_cnn_fw/depthwise_conv2d_6/depthwise_kernelEConvolution2D_DepthWise_0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEonline_cnn_fw/conv2d_6/kernel1Convolution2D_0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
(	variables
)trainable_variables
*regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 
* 
* 
* 
��
VARIABLE_VALUE1online_cnn_fw/depthwise_conv2d_7/depthwise_kernelEConvolution2D_DepthWise_1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*

/0*

/0*
* 
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEonline_cnn_fw/conv2d_7/kernel1Convolution2D_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

60*

60*
* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
* 
* 
|v
VARIABLE_VALUE)online_cnn_fw/batch_normalization_3/gamma9BatchNormalization1D_GAP/gamma/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE(online_cnn_fw/batch_normalization_3/beta8BatchNormalization1D_GAP/beta/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/online_cnn_fw/batch_normalization_3/moving_mean?BatchNormalization1D_GAP/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3online_cnn_fw/batch_normalization_3/moving_varianceCBatchNormalization1D_GAP/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
K0
L1
M2
N3*

K0
L1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 
* 

M0
N1*
R
0
1
2
3
4
5
6
7
	8

9
10*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11online_cnn_fw/depthwise_conv2d_6/depthwise_kernelonline_cnn_fw/conv2d_6/kernel1online_cnn_fw/depthwise_conv2d_7/depthwise_kernelonline_cnn_fw/conv2d_7/kernel3online_cnn_fw/batch_normalization_3/moving_variance)online_cnn_fw/batch_normalization_3/gamma/online_cnn_fw/batch_normalization_3/moving_mean(online_cnn_fw/batch_normalization_3/beta*
Tin
2	*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_41014
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameEonline_cnn_fw/depthwise_conv2d_6/depthwise_kernel/Read/ReadVariableOp1online_cnn_fw/conv2d_6/kernel/Read/ReadVariableOpEonline_cnn_fw/depthwise_conv2d_7/depthwise_kernel/Read/ReadVariableOp1online_cnn_fw/conv2d_7/kernel/Read/ReadVariableOp=online_cnn_fw/batch_normalization_3/gamma/Read/ReadVariableOp<online_cnn_fw/batch_normalization_3/beta/Read/ReadVariableOpConline_cnn_fw/batch_normalization_3/moving_mean/Read/ReadVariableOpGonline_cnn_fw/batch_normalization_3/moving_variance/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_41286
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename1online_cnn_fw/depthwise_conv2d_6/depthwise_kernelonline_cnn_fw/conv2d_6/kernel1online_cnn_fw/depthwise_conv2d_7/depthwise_kernelonline_cnn_fw/conv2d_7/kernel)online_cnn_fw/batch_normalization_3/gamma(online_cnn_fw/batch_normalization_3/beta/online_cnn_fw/batch_normalization_3/moving_mean3online_cnn_fw/batch_normalization_3/moving_variance*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_41320��
�-
�
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40565
x2
depthwise_conv2d_6_40486:(
conv2d_6_40497: 2
depthwise_conv2d_7_40524: (
conv2d_7_40535: @)
batch_normalization_3_40546:@)
batch_normalization_3_40548:@)
batch_normalization_3_40550:@)
batch_normalization_3_40552:@
identity��-batch_normalization_3/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall�*depthwise_conv2d_6/StatefulPartitionedCall�*depthwise_conv2d_7/StatefulPartitionedCall�
*depthwise_conv2d_6/StatefulPartitionedCallStatefulPartitionedCallxdepthwise_conv2d_6_40486*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_40485�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_6/StatefulPartitionedCall:output:0conv2d_6_40497*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_40496�
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_40505�
dropout_6/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_40512�
*depthwise_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0depthwise_conv2d_7_40524*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_40523�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_7/StatefulPartitionedCall:output:0conv2d_7_40535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_40534�
re_lu_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_40543�
*global_average_pooling2d_3/PartitionedCallPartitionedCall re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_40384�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0batch_normalization_3_40546batch_normalization_3_40548batch_normalization_3_40550batch_normalization_3_40552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_40411�
dropout_7/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_40560b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   {
ReshapeReshape"dropout_7/PartitionedCall:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall+^depthwise_conv2d_6/StatefulPartitionedCall+^depthwise_conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2X
*depthwise_conv2d_6/StatefulPartitionedCall*depthwise_conv2d_6/StatefulPartitionedCall2X
*depthwise_conv2d_7/StatefulPartitionedCall*depthwise_conv2d_7/StatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
2__inference_depthwise_conv2d_6_layer_call_fn_41021

inputs!
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_40485w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_40560

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling2d_3_layer_call_fn_41126

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_40384i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�!
�
__inference__traced_save_41286
file_prefixP
Lsavev2_online_cnn_fw_depthwise_conv2d_6_depthwise_kernel_read_readvariableop<
8savev2_online_cnn_fw_conv2d_6_kernel_read_readvariableopP
Lsavev2_online_cnn_fw_depthwise_conv2d_7_depthwise_kernel_read_readvariableop<
8savev2_online_cnn_fw_conv2d_7_kernel_read_readvariableopH
Dsavev2_online_cnn_fw_batch_normalization_3_gamma_read_readvariableopG
Csavev2_online_cnn_fw_batch_normalization_3_beta_read_readvariableopN
Jsavev2_online_cnn_fw_batch_normalization_3_moving_mean_read_readvariableopR
Nsavev2_online_cnn_fw_batch_normalization_3_moving_variance_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	BEConvolution2D_DepthWise_0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB1Convolution2D_0/kernel/.ATTRIBUTES/VARIABLE_VALUEBEConvolution2D_DepthWise_1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB1Convolution2D_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB9BatchNormalization1D_GAP/gamma/.ATTRIBUTES/VARIABLE_VALUEB8BatchNormalization1D_GAP/beta/.ATTRIBUTES/VARIABLE_VALUEB?BatchNormalization1D_GAP/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBCBatchNormalization1D_GAP/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Lsavev2_online_cnn_fw_depthwise_conv2d_6_depthwise_kernel_read_readvariableop8savev2_online_cnn_fw_conv2d_6_kernel_read_readvariableopLsavev2_online_cnn_fw_depthwise_conv2d_7_depthwise_kernel_read_readvariableop8savev2_online_cnn_fw_conv2d_7_kernel_read_readvariableopDsavev2_online_cnn_fw_batch_normalization_3_gamma_read_readvariableopCsavev2_online_cnn_fw_batch_normalization_3_beta_read_readvariableopJsavev2_online_cnn_fw_batch_normalization_3_moving_mean_read_readvariableopNsavev2_online_cnn_fw_batch_normalization_3_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*w
_input_shapesf
d: :: : : @:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
::,(
&
_output_shapes
: :,(
&
_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:	

_output_shapes
: 
�	
�
-__inference_online_cnn_fw_layer_call_fn_40584
input_1!
unknown:#
	unknown_0: #
	unknown_1: #
	unknown_2: @
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40565j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�0
�
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40729
x2
depthwise_conv2d_6_40700:(
conv2d_6_40703: 2
depthwise_conv2d_7_40708: (
conv2d_7_40711: @)
batch_normalization_3_40716:@)
batch_normalization_3_40718:@)
batch_normalization_3_40720:@)
batch_normalization_3_40722:@
identity��-batch_normalization_3/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall�*depthwise_conv2d_6/StatefulPartitionedCall�*depthwise_conv2d_7/StatefulPartitionedCall�!dropout_6/StatefulPartitionedCall�!dropout_7/StatefulPartitionedCall�
*depthwise_conv2d_6/StatefulPartitionedCallStatefulPartitionedCallxdepthwise_conv2d_6_40700*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_40485�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_6/StatefulPartitionedCall:output:0conv2d_6_40703*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_40496�
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_40505�
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_40649�
*depthwise_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0depthwise_conv2d_7_40708*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_40523�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_7/StatefulPartitionedCall:output:0conv2d_7_40711*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_40534�
re_lu_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_40543�
*global_average_pooling2d_3/PartitionedCallPartitionedCall re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_40384�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0batch_normalization_3_40716batch_normalization_3_40718batch_normalization_3_40720batch_normalization_3_40722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_40458�
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_40604b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   �
ReshapeReshape*dropout_7/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall+^depthwise_conv2d_6/StatefulPartitionedCall+^depthwise_conv2d_7/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2X
*depthwise_conv2d_6/StatefulPartitionedCall*depthwise_conv2d_6/StatefulPartitionedCall2X
*depthwise_conv2d_7/StatefulPartitionedCall*depthwise_conv2d_7/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_41044

inputs8
conv2d_readvariableop_resource: 
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:��������� ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
)__inference_dropout_7_layer_call_fn_41222

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_40604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_41030

inputs;
!depthwise_readvariableop_resource:
identity��depthwise/ReadVariableOp�
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
i
IdentityIdentitydepthwise:output:0^NoOp*
T0*/
_output_shapes
:���������a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������  : 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�	
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_41239

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�0
�
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40833
input_12
depthwise_conv2d_6_40804:(
conv2d_6_40807: 2
depthwise_conv2d_7_40812: (
conv2d_7_40815: @)
batch_normalization_3_40820:@)
batch_normalization_3_40822:@)
batch_normalization_3_40824:@)
batch_normalization_3_40826:@
identity��-batch_normalization_3/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall�*depthwise_conv2d_6/StatefulPartitionedCall�*depthwise_conv2d_7/StatefulPartitionedCall�!dropout_6/StatefulPartitionedCall�!dropout_7/StatefulPartitionedCall�
*depthwise_conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_1depthwise_conv2d_6_40804*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_40485�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_6/StatefulPartitionedCall:output:0conv2d_6_40807*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_40496�
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_40505�
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_40649�
*depthwise_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0depthwise_conv2d_7_40812*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_40523�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_7/StatefulPartitionedCall:output:0conv2d_7_40815*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_40534�
re_lu_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_40543�
*global_average_pooling2d_3/PartitionedCallPartitionedCall re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_40384�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0batch_normalization_3_40820batch_normalization_3_40822batch_normalization_3_40824batch_normalization_3_40826*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_40458�
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_40604b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   �
ReshapeReshape*dropout_7/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall+^depthwise_conv2d_6/StatefulPartitionedCall+^depthwise_conv2d_7/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2X
*depthwise_conv2d_6/StatefulPartitionedCall*depthwise_conv2d_6/StatefulPartitionedCall2X
*depthwise_conv2d_7/StatefulPartitionedCall*depthwise_conv2d_7/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
�
(__inference_conv2d_6_layer_call_fn_41037

inputs!
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_40496w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_40604

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_41132

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_40485

inputs;
!depthwise_readvariableop_resource:
identity��depthwise/ReadVariableOp�
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
i
IdentityIdentitydepthwise:output:0^NoOp*
T0*/
_output_shapes
:���������a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������  : 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�:
�
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40919
xN
4depthwise_conv2d_6_depthwise_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource: N
4depthwise_conv2d_7_depthwise_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: @E
7batch_normalization_3_batchnorm_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_3_batchnorm_readvariableop_1_resource:@G
9batch_normalization_3_batchnorm_readvariableop_2_resource:@
identity��.batch_normalization_3/batchnorm/ReadVariableOp�0batch_normalization_3/batchnorm/ReadVariableOp_1�0batch_normalization_3/batchnorm/ReadVariableOp_2�2batch_normalization_3/batchnorm/mul/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�+depthwise_conv2d_6/depthwise/ReadVariableOp�+depthwise_conv2d_7/depthwise/ReadVariableOp�
+depthwise_conv2d_6/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_6_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0{
"depthwise_conv2d_6/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            {
*depthwise_conv2d_6/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_6/depthwiseDepthwiseConv2dNativex3depthwise_conv2d_6/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/Conv2DConv2D%depthwise_conv2d_6/depthwise:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
h
re_lu_6/ReluReluconv2d_6/Conv2D:output:0*
T0*/
_output_shapes
:��������� t
dropout_6/IdentityIdentityre_lu_6/Relu:activations:0*
T0*/
_output_shapes
:��������� �
+depthwise_conv2d_7/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_7_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0{
"depthwise_conv2d_7/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             {
*depthwise_conv2d_7/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_7/depthwiseDepthwiseConv2dNativedropout_6/Identity:output:03depthwise_conv2d_7/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_7/Conv2DConv2D%depthwise_conv2d_7/depthwise:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
h
re_lu_7/ReluReluconv2d_7/Conv2D:output:0*
T0*/
_output_shapes
:���������@�
1global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
global_average_pooling2d_3/MeanMeanre_lu_7/Relu:activations:0:global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_3/batchnorm/mul_1Mul(global_average_pooling2d_3/Mean:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@{
dropout_7/IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   t
ReshapeReshapedropout_7/Identity:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp,^depthwise_conv2d_6/depthwise/ReadVariableOp,^depthwise_conv2d_7/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2Z
+depthwise_conv2d_6/depthwise/ReadVariableOp+depthwise_conv2d_6/depthwise/ReadVariableOp2Z
+depthwise_conv2d_7/depthwise/ReadVariableOp+depthwise_conv2d_7/depthwise/ReadVariableOp:R N
/
_output_shapes
:���������  

_user_specified_namex
�
^
B__inference_re_lu_6_layer_call_and_return_conditional_losses_40505

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:��������� b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
b
)__inference_dropout_6_layer_call_fn_41064

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_40649w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
2__inference_depthwise_conv2d_7_layer_call_fn_41088

inputs!
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_40523w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_41227

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�(
�
!__inference__traced_restore_41320
file_prefix\
Bassignvariableop_online_cnn_fw_depthwise_conv2d_6_depthwise_kernel:J
0assignvariableop_1_online_cnn_fw_conv2d_6_kernel: ^
Dassignvariableop_2_online_cnn_fw_depthwise_conv2d_7_depthwise_kernel: J
0assignvariableop_3_online_cnn_fw_conv2d_7_kernel: @J
<assignvariableop_4_online_cnn_fw_batch_normalization_3_gamma:@I
;assignvariableop_5_online_cnn_fw_batch_normalization_3_beta:@P
Bassignvariableop_6_online_cnn_fw_batch_normalization_3_moving_mean:@T
Fassignvariableop_7_online_cnn_fw_batch_normalization_3_moving_variance:@

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	BEConvolution2D_DepthWise_0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB1Convolution2D_0/kernel/.ATTRIBUTES/VARIABLE_VALUEBEConvolution2D_DepthWise_1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB1Convolution2D_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB9BatchNormalization1D_GAP/gamma/.ATTRIBUTES/VARIABLE_VALUEB8BatchNormalization1D_GAP/beta/.ATTRIBUTES/VARIABLE_VALUEB?BatchNormalization1D_GAP/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBCBatchNormalization1D_GAP/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpBassignvariableop_online_cnn_fw_depthwise_conv2d_6_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp0assignvariableop_1_online_cnn_fw_conv2d_6_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpDassignvariableop_2_online_cnn_fw_depthwise_conv2d_7_depthwise_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_online_cnn_fw_conv2d_7_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp<assignvariableop_4_online_cnn_fw_batch_normalization_3_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp;assignvariableop_5_online_cnn_fw_batch_normalization_3_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpBassignvariableop_6_online_cnn_fw_batch_normalization_3_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpFassignvariableop_7_online_cnn_fw_batch_normalization_3_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41178

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
C
'__inference_re_lu_6_layer_call_fn_41049

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_40505h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
C
'__inference_re_lu_7_layer_call_fn_41116

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_40543h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
^
B__inference_re_lu_7_layer_call_and_return_conditional_losses_40543

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
-__inference_online_cnn_fw_layer_call_fn_40769
input_1!
unknown:#
	unknown_0: #
	unknown_1: #
	unknown_2: @
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40729j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�F
�	
 __inference__wrapped_model_40374
input_1\
Bonline_cnn_fw_depthwise_conv2d_6_depthwise_readvariableop_resource:O
5online_cnn_fw_conv2d_6_conv2d_readvariableop_resource: \
Bonline_cnn_fw_depthwise_conv2d_7_depthwise_readvariableop_resource: O
5online_cnn_fw_conv2d_7_conv2d_readvariableop_resource: @S
Eonline_cnn_fw_batch_normalization_3_batchnorm_readvariableop_resource:@W
Ionline_cnn_fw_batch_normalization_3_batchnorm_mul_readvariableop_resource:@U
Gonline_cnn_fw_batch_normalization_3_batchnorm_readvariableop_1_resource:@U
Gonline_cnn_fw_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identity��<online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp�>online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_1�>online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_2�@online_cnn_fw/batch_normalization_3/batchnorm/mul/ReadVariableOp�,online_cnn_fw/conv2d_6/Conv2D/ReadVariableOp�,online_cnn_fw/conv2d_7/Conv2D/ReadVariableOp�9online_cnn_fw/depthwise_conv2d_6/depthwise/ReadVariableOp�9online_cnn_fw/depthwise_conv2d_7/depthwise/ReadVariableOp�
9online_cnn_fw/depthwise_conv2d_6/depthwise/ReadVariableOpReadVariableOpBonline_cnn_fw_depthwise_conv2d_6_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0�
0online_cnn_fw/depthwise_conv2d_6/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
8online_cnn_fw/depthwise_conv2d_6/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
*online_cnn_fw/depthwise_conv2d_6/depthwiseDepthwiseConv2dNativeinput_1Aonline_cnn_fw/depthwise_conv2d_6/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
,online_cnn_fw/conv2d_6/Conv2D/ReadVariableOpReadVariableOp5online_cnn_fw_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
online_cnn_fw/conv2d_6/Conv2DConv2D3online_cnn_fw/depthwise_conv2d_6/depthwise:output:04online_cnn_fw/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
online_cnn_fw/re_lu_6/ReluRelu&online_cnn_fw/conv2d_6/Conv2D:output:0*
T0*/
_output_shapes
:��������� �
 online_cnn_fw/dropout_6/IdentityIdentity(online_cnn_fw/re_lu_6/Relu:activations:0*
T0*/
_output_shapes
:��������� �
9online_cnn_fw/depthwise_conv2d_7/depthwise/ReadVariableOpReadVariableOpBonline_cnn_fw_depthwise_conv2d_7_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0�
0online_cnn_fw/depthwise_conv2d_7/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
8online_cnn_fw/depthwise_conv2d_7/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
*online_cnn_fw/depthwise_conv2d_7/depthwiseDepthwiseConv2dNative)online_cnn_fw/dropout_6/Identity:output:0Aonline_cnn_fw/depthwise_conv2d_7/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,online_cnn_fw/conv2d_7/Conv2D/ReadVariableOpReadVariableOp5online_cnn_fw_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
online_cnn_fw/conv2d_7/Conv2DConv2D3online_cnn_fw/depthwise_conv2d_7/depthwise:output:04online_cnn_fw/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
online_cnn_fw/re_lu_7/ReluRelu&online_cnn_fw/conv2d_7/Conv2D:output:0*
T0*/
_output_shapes
:���������@�
?online_cnn_fw/global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
-online_cnn_fw/global_average_pooling2d_3/MeanMean(online_cnn_fw/re_lu_7/Relu:activations:0Honline_cnn_fw/global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@�
<online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpEonline_cnn_fw_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0x
3online_cnn_fw/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1online_cnn_fw/batch_normalization_3/batchnorm/addAddV2Donline_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp:value:0<online_cnn_fw/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
3online_cnn_fw/batch_normalization_3/batchnorm/RsqrtRsqrt5online_cnn_fw/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@�
@online_cnn_fw/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpIonline_cnn_fw_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
1online_cnn_fw/batch_normalization_3/batchnorm/mulMul7online_cnn_fw/batch_normalization_3/batchnorm/Rsqrt:y:0Honline_cnn_fw/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
3online_cnn_fw/batch_normalization_3/batchnorm/mul_1Mul6online_cnn_fw/global_average_pooling2d_3/Mean:output:05online_cnn_fw/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
>online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpGonline_cnn_fw_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3online_cnn_fw/batch_normalization_3/batchnorm/mul_2MulFonline_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_1:value:05online_cnn_fw/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
>online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpGonline_cnn_fw_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
1online_cnn_fw/batch_normalization_3/batchnorm/subSubFonline_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_2:value:07online_cnn_fw/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
3online_cnn_fw/batch_normalization_3/batchnorm/add_1AddV27online_cnn_fw/batch_normalization_3/batchnorm/mul_1:z:05online_cnn_fw/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
 online_cnn_fw/dropout_7/IdentityIdentity7online_cnn_fw/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@p
online_cnn_fw/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   �
online_cnn_fw/ReshapeReshape)online_cnn_fw/dropout_7/Identity:output:0$online_cnn_fw/Reshape/shape:output:0*
T0*"
_output_shapes
:@h
IdentityIdentityonline_cnn_fw/Reshape:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp=^online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp?^online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_1?^online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_2A^online_cnn_fw/batch_normalization_3/batchnorm/mul/ReadVariableOp-^online_cnn_fw/conv2d_6/Conv2D/ReadVariableOp-^online_cnn_fw/conv2d_7/Conv2D/ReadVariableOp:^online_cnn_fw/depthwise_conv2d_6/depthwise/ReadVariableOp:^online_cnn_fw/depthwise_conv2d_7/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2|
<online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp<online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp2�
>online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_1>online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_12�
>online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_2>online_cnn_fw/batch_normalization_3/batchnorm/ReadVariableOp_22�
@online_cnn_fw/batch_normalization_3/batchnorm/mul/ReadVariableOp@online_cnn_fw/batch_normalization_3/batchnorm/mul/ReadVariableOp2\
,online_cnn_fw/conv2d_6/Conv2D/ReadVariableOp,online_cnn_fw/conv2d_6/Conv2D/ReadVariableOp2\
,online_cnn_fw/conv2d_7/Conv2D/ReadVariableOp,online_cnn_fw/conv2d_7/Conv2D/ReadVariableOp2v
9online_cnn_fw/depthwise_conv2d_6/depthwise/ReadVariableOp9online_cnn_fw/depthwise_conv2d_6/depthwise/ReadVariableOp2v
9online_cnn_fw/depthwise_conv2d_7/depthwise/ReadVariableOp9online_cnn_fw/depthwise_conv2d_7/depthwise/ReadVariableOp:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�

c
D__inference_dropout_6_layer_call_and_return_conditional_losses_40649

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_40411

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_conv2d_7_layer_call_fn_41104

inputs!
unknown: @
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_40534w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
-__inference_online_cnn_fw_layer_call_fn_40875
x!
unknown:#
	unknown_0: #
	unknown_1: #
	unknown_2: @
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40729j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
^
B__inference_re_lu_6_layer_call_and_return_conditional_losses_41054

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:��������� b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�-
�
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40801
input_12
depthwise_conv2d_6_40772:(
conv2d_6_40775: 2
depthwise_conv2d_7_40780: (
conv2d_7_40783: @)
batch_normalization_3_40788:@)
batch_normalization_3_40790:@)
batch_normalization_3_40792:@)
batch_normalization_3_40794:@
identity��-batch_normalization_3/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall�*depthwise_conv2d_6/StatefulPartitionedCall�*depthwise_conv2d_7/StatefulPartitionedCall�
*depthwise_conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_1depthwise_conv2d_6_40772*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_40485�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_6/StatefulPartitionedCall:output:0conv2d_6_40775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_40496�
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_40505�
dropout_6/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_40512�
*depthwise_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0depthwise_conv2d_7_40780*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_40523�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_7/StatefulPartitionedCall:output:0conv2d_7_40783*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_40534�
re_lu_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_40543�
*global_average_pooling2d_3/PartitionedCallPartitionedCall re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_40384�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0batch_normalization_3_40788batch_normalization_3_40790batch_normalization_3_40792batch_normalization_3_40794*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_40411�
dropout_7/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_40560b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   {
ReshapeReshape"dropout_7/PartitionedCall:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall+^depthwise_conv2d_6/StatefulPartitionedCall+^depthwise_conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2X
*depthwise_conv2d_6/StatefulPartitionedCall*depthwise_conv2d_6/StatefulPartitionedCall2X
*depthwise_conv2d_7/StatefulPartitionedCall*depthwise_conv2d_7/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�

c
D__inference_dropout_6_layer_call_and_return_conditional_losses_41081

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_41097

inputs;
!depthwise_readvariableop_resource: 
identity��depthwise/ReadVariableOp�
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
i
IdentityIdentitydepthwise:output:0^NoOp*
T0*/
_output_shapes
:��������� a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_40523

inputs;
!depthwise_readvariableop_resource: 
identity��depthwise/ReadVariableOp�
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
i
IdentityIdentitydepthwise:output:0^NoOp*
T0*/
_output_shapes
:��������� a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
#__inference_signature_wrapper_41014
input_1!
unknown:#
	unknown_0: #
	unknown_1: #
	unknown_2: @
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_40374j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_40534

inputs8
conv2d_readvariableop_resource: @
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:���������@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_3_layer_call_fn_41145

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_40411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�%
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_40458

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
^
B__inference_re_lu_7_layer_call_and_return_conditional_losses_41121

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_40384

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_40496

inputs8
conv2d_readvariableop_resource: 
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:��������� ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_40512

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_41069

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
-__inference_online_cnn_fw_layer_call_fn_40854
x!
unknown:#
	unknown_0: #
	unknown_1: #
	unknown_2: @
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40565j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
E
)__inference_dropout_7_layer_call_fn_41217

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_40560`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
E
)__inference_dropout_6_layer_call_fn_41059

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_40512h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�c
�
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40991
xN
4depthwise_conv2d_6_depthwise_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource: N
4depthwise_conv2d_7_depthwise_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: @K
=batch_normalization_3_assignmovingavg_readvariableop_resource:@M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@
identity��%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�2batch_normalization_3/batchnorm/mul/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�+depthwise_conv2d_6/depthwise/ReadVariableOp�+depthwise_conv2d_7/depthwise/ReadVariableOp�
+depthwise_conv2d_6/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_6_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0{
"depthwise_conv2d_6/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            {
*depthwise_conv2d_6/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_6/depthwiseDepthwiseConv2dNativex3depthwise_conv2d_6/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/Conv2DConv2D%depthwise_conv2d_6/depthwise:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
h
re_lu_6/ReluReluconv2d_6/Conv2D:output:0*
T0*/
_output_shapes
:��������� \
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_6/dropout/MulMulre_lu_6/Relu:activations:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:��������� a
dropout_6/dropout/ShapeShapere_lu_6/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;�
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� �
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� �
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:��������� �
+depthwise_conv2d_7/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_7_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0{
"depthwise_conv2d_7/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             {
*depthwise_conv2d_7/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_7/depthwiseDepthwiseConv2dNativedropout_6/dropout/Mul_1:z:03depthwise_conv2d_7/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_7/Conv2DConv2D%depthwise_conv2d_7/depthwise:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
h
re_lu_7/ReluReluconv2d_7/Conv2D:output:0*
T0*/
_output_shapes
:���������@�
1global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
global_average_pooling2d_3/MeanMeanre_lu_7/Relu:activations:0:global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_3/moments/meanMean(global_average_pooling2d_3/Mean:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifference(global_average_pooling2d_3/Mean:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_3/batchnorm/mul_1Mul(global_average_pooling2d_3/Mean:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_7/dropout/MulMul)batch_normalization_3/batchnorm/add_1:z:0 dropout_7/dropout/Const:output:0*
T0*'
_output_shapes
:���������@p
dropout_7/dropout/ShapeShape)batch_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   t
ReshapeReshapedropout_7/dropout/Mul_1:z:0Reshape/shape:output:0*
T0*"
_output_shapes
:@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp,^depthwise_conv2d_6/depthwise/ReadVariableOp,^depthwise_conv2d_7/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2Z
+depthwise_conv2d_6/depthwise/ReadVariableOp+depthwise_conv2d_6/depthwise/ReadVariableOp2Z
+depthwise_conv2d_7/depthwise/ReadVariableOp+depthwise_conv2d_7/depthwise/ReadVariableOp:R N
/
_output_shapes
:���������  

_user_specified_namex
�%
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41212

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_3_layer_call_fn_41158

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_40458o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_41111

inputs8
conv2d_readvariableop_resource: @
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:���������@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������  7
output_1+
StatefulPartitionedCall:0@tensorflow/serving/predict:۫
�
Convolution2D_DepthWise_0
Convolution2D_0

ReLU_0
	Dropout_0
Convolution2D_DepthWise_1
Convolution2D_1

ReLU_1
GlobalAveragePool2D_GAP
	Flatten_GAP

BatchNormalization1D_GAP
Dropout_GAP
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
�
depthwise_kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,_random_generator
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/depthwise_kernel
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
�

6kernel
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
(
I	keras_api"
_tf_keras_layer
�
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y_random_generator
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
X
0
1
/2
63
K4
L5
M6
N7"
trackable_list_wrapper
J
0
1
/2
63
K4
L5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_online_cnn_fw_layer_call_fn_40584
-__inference_online_cnn_fw_layer_call_fn_40854
-__inference_online_cnn_fw_layer_call_fn_40875
-__inference_online_cnn_fw_layer_call_fn_40769�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40919
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40991
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40801
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40833�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
 __inference__wrapped_model_40374input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
aserving_default"
signature_map
K:I21online_cnn_fw/depthwise_conv2d_6/depthwise_kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_depthwise_conv2d_6_layer_call_fn_41021�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_41030�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
7:5 2online_cnn_fw/conv2d_6/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_conv2d_6_layer_call_fn_41037�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_41044�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_re_lu_6_layer_call_fn_41049�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_re_lu_6_layer_call_and_return_conditional_losses_41054�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
(	variables
)trainable_variables
*regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
)__inference_dropout_6_layer_call_fn_41059
)__inference_dropout_6_layer_call_fn_41064�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dropout_6_layer_call_and_return_conditional_losses_41069
D__inference_dropout_6_layer_call_and_return_conditional_losses_41081�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
K:I 21online_cnn_fw/depthwise_conv2d_7/depthwise_kernel
'
/0"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_depthwise_conv2d_7_layer_call_fn_41088�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_41097�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
7:5 @2online_cnn_fw/conv2d_7/kernel
'
60"
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_conv2d_7_layer_call_fn_41104�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_41111�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_re_lu_7_layer_call_fn_41116�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_re_lu_7_layer_call_and_return_conditional_losses_41121�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�2�
:__inference_global_average_pooling2d_3_layer_call_fn_41126�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_41132�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
7:5@2)online_cnn_fw/batch_normalization_3/gamma
6:4@2(online_cnn_fw/batch_normalization_3/beta
?:=@ (2/online_cnn_fw/batch_normalization_3/moving_mean
C:A@ (23online_cnn_fw/batch_normalization_3/moving_variance
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_batch_normalization_3_layer_call_fn_41145
5__inference_batch_normalization_3_layer_call_fn_41158�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41178
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41212�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
)__inference_dropout_7_layer_call_fn_41217
)__inference_dropout_7_layer_call_fn_41222�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dropout_7_layer_call_and_return_conditional_losses_41227
D__inference_dropout_7_layer_call_and_return_conditional_losses_41239�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
.
M0
N1"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_signature_wrapper_41014input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
 __inference__wrapped_model_40374t/6NKML8�5
.�+
)�&
input_1���������  
� ".�+
)
output_1�
output_1@�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41178bNKML3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41212bMNKL3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
5__inference_batch_normalization_3_layer_call_fn_41145UNKML3�0
)�&
 �
inputs���������@
p 
� "����������@�
5__inference_batch_normalization_3_layer_call_fn_41158UMNKL3�0
)�&
 �
inputs���������@
p
� "����������@�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_41044k7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
(__inference_conv2d_6_layer_call_fn_41037^7�4
-�*
(�%
inputs���������
� " ���������� �
C__inference_conv2d_7_layer_call_and_return_conditional_losses_41111k67�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
(__inference_conv2d_7_layer_call_fn_41104^67�4
-�*
(�%
inputs��������� 
� " ����������@�
M__inference_depthwise_conv2d_6_layer_call_and_return_conditional_losses_41030k7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������
� �
2__inference_depthwise_conv2d_6_layer_call_fn_41021^7�4
-�*
(�%
inputs���������  
� " �����������
M__inference_depthwise_conv2d_7_layer_call_and_return_conditional_losses_41097k/7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0��������� 
� �
2__inference_depthwise_conv2d_7_layer_call_fn_41088^/7�4
-�*
(�%
inputs��������� 
� " ���������� �
D__inference_dropout_6_layer_call_and_return_conditional_losses_41069l;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
D__inference_dropout_6_layer_call_and_return_conditional_losses_41081l;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
)__inference_dropout_6_layer_call_fn_41059_;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
)__inference_dropout_6_layer_call_fn_41064_;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
D__inference_dropout_7_layer_call_and_return_conditional_losses_41227\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
D__inference_dropout_7_layer_call_and_return_conditional_losses_41239\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� |
)__inference_dropout_7_layer_call_fn_41217O3�0
)�&
 �
inputs���������@
p 
� "����������@|
)__inference_dropout_7_layer_call_fn_41222O3�0
)�&
 �
inputs���������@
p
� "����������@�
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_41132�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0������������������
� �
:__inference_global_average_pooling2d_3_layer_call_fn_41126wR�O
H�E
C�@
inputs4������������������������������������
� "!��������������������
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40801j/6NKML<�9
2�/
)�&
input_1���������  
p 
� " �
�
0@
� �
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40833j/6MNKL<�9
2�/
)�&
input_1���������  
p
� " �
�
0@
� �
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40919d/6NKML6�3
,�)
#� 
x���������  
p 
� " �
�
0@
� �
H__inference_online_cnn_fw_layer_call_and_return_conditional_losses_40991d/6MNKL6�3
,�)
#� 
x���������  
p
� " �
�
0@
� �
-__inference_online_cnn_fw_layer_call_fn_40584]/6NKML<�9
2�/
)�&
input_1���������  
p 
� "�@�
-__inference_online_cnn_fw_layer_call_fn_40769]/6MNKL<�9
2�/
)�&
input_1���������  
p
� "�@�
-__inference_online_cnn_fw_layer_call_fn_40854W/6NKML6�3
,�)
#� 
x���������  
p 
� "�@�
-__inference_online_cnn_fw_layer_call_fn_40875W/6MNKL6�3
,�)
#� 
x���������  
p
� "�@�
B__inference_re_lu_6_layer_call_and_return_conditional_losses_41054h7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0��������� 
� �
'__inference_re_lu_6_layer_call_fn_41049[7�4
-�*
(�%
inputs��������� 
� " ���������� �
B__inference_re_lu_7_layer_call_and_return_conditional_losses_41121h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
'__inference_re_lu_7_layer_call_fn_41116[7�4
-�*
(�%
inputs���������@
� " ����������@�
#__inference_signature_wrapper_41014/6NKMLC�@
� 
9�6
4
input_1)�&
input_1���������  ".�+
)
output_1�
output_1@
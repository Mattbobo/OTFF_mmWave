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
0online_cnn_1/depthwise_conv2d_4/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20online_cnn_1/depthwise_conv2d_4/depthwise_kernel
�
Donline_cnn_1/depthwise_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp0online_cnn_1/depthwise_conv2d_4/depthwise_kernel*&
_output_shapes
:*
dtype0
�
online_cnn_1/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameonline_cnn_1/conv2d_4/kernel
�
0online_cnn_1/conv2d_4/kernel/Read/ReadVariableOpReadVariableOponline_cnn_1/conv2d_4/kernel*&
_output_shapes
: *
dtype0
�
0online_cnn_1/depthwise_conv2d_5/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20online_cnn_1/depthwise_conv2d_5/depthwise_kernel
�
Donline_cnn_1/depthwise_conv2d_5/depthwise_kernel/Read/ReadVariableOpReadVariableOp0online_cnn_1/depthwise_conv2d_5/depthwise_kernel*&
_output_shapes
: *
dtype0
�
online_cnn_1/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*-
shared_nameonline_cnn_1/conv2d_5/kernel
�
0online_cnn_1/conv2d_5/kernel/Read/ReadVariableOpReadVariableOponline_cnn_1/conv2d_5/kernel*&
_output_shapes
: @*
dtype0
�
(online_cnn_1/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(online_cnn_1/batch_normalization_2/gamma
�
<online_cnn_1/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp(online_cnn_1/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
�
'online_cnn_1/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'online_cnn_1/batch_normalization_2/beta
�
;online_cnn_1/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp'online_cnn_1/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
�
.online_cnn_1/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.online_cnn_1/batch_normalization_2/moving_mean
�
Bonline_cnn_1/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp.online_cnn_1/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
�
2online_cnn_1/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42online_cnn_1/batch_normalization_2/moving_variance
�
Fonline_cnn_1/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp2online_cnn_1/batch_normalization_2/moving_variance*
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
VARIABLE_VALUE0online_cnn_1/depthwise_conv2d_4/depthwise_kernelEConvolution2D_DepthWise_0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*

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
ga
VARIABLE_VALUEonline_cnn_1/conv2d_4/kernel1Convolution2D_0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

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
VARIABLE_VALUE0online_cnn_1/depthwise_conv2d_5/depthwise_kernelEConvolution2D_DepthWise_1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*

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
ga
VARIABLE_VALUEonline_cnn_1/conv2d_5/kernel1Convolution2D_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

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
{u
VARIABLE_VALUE(online_cnn_1/batch_normalization_2/gamma9BatchNormalization1D_GAP/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE'online_cnn_1/batch_normalization_2/beta8BatchNormalization1D_GAP/beta/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE.online_cnn_1/batch_normalization_2/moving_mean?BatchNormalization1D_GAP/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE2online_cnn_1/batch_normalization_2/moving_varianceCBatchNormalization1D_GAP/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10online_cnn_1/depthwise_conv2d_4/depthwise_kernelonline_cnn_1/conv2d_4/kernel0online_cnn_1/depthwise_conv2d_5/depthwise_kernelonline_cnn_1/conv2d_5/kernel2online_cnn_1/batch_normalization_2/moving_variance(online_cnn_1/batch_normalization_2/gamma.online_cnn_1/batch_normalization_2/moving_mean'online_cnn_1/batch_normalization_2/beta*
Tin
2	*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:#@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_39976
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameDonline_cnn_1/depthwise_conv2d_4/depthwise_kernel/Read/ReadVariableOp0online_cnn_1/conv2d_4/kernel/Read/ReadVariableOpDonline_cnn_1/depthwise_conv2d_5/depthwise_kernel/Read/ReadVariableOp0online_cnn_1/conv2d_5/kernel/Read/ReadVariableOp<online_cnn_1/batch_normalization_2/gamma/Read/ReadVariableOp;online_cnn_1/batch_normalization_2/beta/Read/ReadVariableOpBonline_cnn_1/batch_normalization_2/moving_mean/Read/ReadVariableOpFonline_cnn_1/batch_normalization_2/moving_variance/Read/ReadVariableOpConst*
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
__inference__traced_save_40248
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename0online_cnn_1/depthwise_conv2d_4/depthwise_kernelonline_cnn_1/conv2d_4/kernel0online_cnn_1/depthwise_conv2d_5/depthwise_kernelonline_cnn_1/conv2d_5/kernel(online_cnn_1/batch_normalization_2/gamma'online_cnn_1/batch_normalization_2/beta.online_cnn_1/batch_normalization_2/moving_mean2online_cnn_1/batch_normalization_2/moving_variance*
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
!__inference__traced_restore_40282��
�	
�
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39447

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
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40006

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
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_39467

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
�	
�
,__inference_online_cnn_1_layer_call_fn_39816
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
:#@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39527j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:#@`
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
B__inference_re_lu_4_layer_call_and_return_conditional_losses_40016

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
�	
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_39566

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
�
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_40189

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
�

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_40043

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
�
E
)__inference_dropout_5_layer_call_fn_40179

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
D__inference_dropout_5_layer_call_and_return_conditional_losses_39522`
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
�-
�
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39527
x2
depthwise_conv2d_4_39448:(
conv2d_4_39459: 2
depthwise_conv2d_5_39486: (
conv2d_5_39497: @)
batch_normalization_2_39508:@)
batch_normalization_2_39510:@)
batch_normalization_2_39512:@)
batch_normalization_2_39514:@
identity��-batch_normalization_2/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�*depthwise_conv2d_4/StatefulPartitionedCall�*depthwise_conv2d_5/StatefulPartitionedCall�
*depthwise_conv2d_4/StatefulPartitionedCallStatefulPartitionedCallxdepthwise_conv2d_4_39448*
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
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39447�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_4/StatefulPartitionedCall:output:0conv2d_4_39459*
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_39458�
re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
B__inference_re_lu_4_layer_call_and_return_conditional_losses_39467�
dropout_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
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
D__inference_dropout_4_layer_call_and_return_conditional_losses_39474�
*depthwise_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0depthwise_conv2d_5_39486*
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
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_39485�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_5/StatefulPartitionedCall:output:0conv2d_5_39497*
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_39496�
re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
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
B__inference_re_lu_5_layer_call_and_return_conditional_losses_39505�
*global_average_pooling2d_2/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_39346�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0batch_normalization_2_39508batch_normalization_2_39510batch_normalization_2_39512batch_normalization_2_39514*
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_39373�
dropout_5/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_39522b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   #   @   {
ReshapeReshape"dropout_5/PartitionedCall:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:#@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:#@�
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall+^depthwise_conv2d_4/StatefulPartitionedCall+^depthwise_conv2d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2X
*depthwise_conv2d_4/StatefulPartitionedCall*depthwise_conv2d_4/StatefulPartitionedCall2X
*depthwise_conv2d_5/StatefulPartitionedCall*depthwise_conv2d_5/StatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�E
�	
 __inference__wrapped_model_39336
input_1[
Aonline_cnn_1_depthwise_conv2d_4_depthwise_readvariableop_resource:N
4online_cnn_1_conv2d_4_conv2d_readvariableop_resource: [
Aonline_cnn_1_depthwise_conv2d_5_depthwise_readvariableop_resource: N
4online_cnn_1_conv2d_5_conv2d_readvariableop_resource: @R
Donline_cnn_1_batch_normalization_2_batchnorm_readvariableop_resource:@V
Honline_cnn_1_batch_normalization_2_batchnorm_mul_readvariableop_resource:@T
Fonline_cnn_1_batch_normalization_2_batchnorm_readvariableop_1_resource:@T
Fonline_cnn_1_batch_normalization_2_batchnorm_readvariableop_2_resource:@
identity��;online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp�=online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_1�=online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_2�?online_cnn_1/batch_normalization_2/batchnorm/mul/ReadVariableOp�+online_cnn_1/conv2d_4/Conv2D/ReadVariableOp�+online_cnn_1/conv2d_5/Conv2D/ReadVariableOp�8online_cnn_1/depthwise_conv2d_4/depthwise/ReadVariableOp�8online_cnn_1/depthwise_conv2d_5/depthwise/ReadVariableOp�
8online_cnn_1/depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOpAonline_cnn_1_depthwise_conv2d_4_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0�
/online_cnn_1/depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
7online_cnn_1/depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
)online_cnn_1/depthwise_conv2d_4/depthwiseDepthwiseConv2dNativeinput_1@online_cnn_1/depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
+online_cnn_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4online_cnn_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
online_cnn_1/conv2d_4/Conv2DConv2D2online_cnn_1/depthwise_conv2d_4/depthwise:output:03online_cnn_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
online_cnn_1/re_lu_4/ReluRelu%online_cnn_1/conv2d_4/Conv2D:output:0*
T0*/
_output_shapes
:��������� �
online_cnn_1/dropout_4/IdentityIdentity'online_cnn_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:��������� �
8online_cnn_1/depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOpAonline_cnn_1_depthwise_conv2d_5_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0�
/online_cnn_1/depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
7online_cnn_1/depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
)online_cnn_1/depthwise_conv2d_5/depthwiseDepthwiseConv2dNative(online_cnn_1/dropout_4/Identity:output:0@online_cnn_1/depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
+online_cnn_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4online_cnn_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
online_cnn_1/conv2d_5/Conv2DConv2D2online_cnn_1/depthwise_conv2d_5/depthwise:output:03online_cnn_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
online_cnn_1/re_lu_5/ReluRelu%online_cnn_1/conv2d_5/Conv2D:output:0*
T0*/
_output_shapes
:���������@�
>online_cnn_1/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
,online_cnn_1/global_average_pooling2d_2/MeanMean'online_cnn_1/re_lu_5/Relu:activations:0Gonline_cnn_1/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@�
;online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpDonline_cnn_1_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0w
2online_cnn_1/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0online_cnn_1/batch_normalization_2/batchnorm/addAddV2Conline_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp:value:0;online_cnn_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
2online_cnn_1/batch_normalization_2/batchnorm/RsqrtRsqrt4online_cnn_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
?online_cnn_1/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHonline_cnn_1_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
0online_cnn_1/batch_normalization_2/batchnorm/mulMul6online_cnn_1/batch_normalization_2/batchnorm/Rsqrt:y:0Gonline_cnn_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
2online_cnn_1/batch_normalization_2/batchnorm/mul_1Mul5online_cnn_1/global_average_pooling2d_2/Mean:output:04online_cnn_1/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
=online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpFonline_cnn_1_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
2online_cnn_1/batch_normalization_2/batchnorm/mul_2MulEonline_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_1:value:04online_cnn_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
=online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpFonline_cnn_1_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
0online_cnn_1/batch_normalization_2/batchnorm/subSubEonline_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_2:value:06online_cnn_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
2online_cnn_1/batch_normalization_2/batchnorm/add_1AddV26online_cnn_1/batch_normalization_2/batchnorm/mul_1:z:04online_cnn_1/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
online_cnn_1/dropout_5/IdentityIdentity6online_cnn_1/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@o
online_cnn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   #   @   �
online_cnn_1/ReshapeReshape(online_cnn_1/dropout_5/Identity:output:0#online_cnn_1/Reshape/shape:output:0*
T0*"
_output_shapes
:#@g
IdentityIdentityonline_cnn_1/Reshape:output:0^NoOp*
T0*"
_output_shapes
:#@�
NoOpNoOp<^online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp>^online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_1>^online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_2@^online_cnn_1/batch_normalization_2/batchnorm/mul/ReadVariableOp,^online_cnn_1/conv2d_4/Conv2D/ReadVariableOp,^online_cnn_1/conv2d_5/Conv2D/ReadVariableOp9^online_cnn_1/depthwise_conv2d_4/depthwise/ReadVariableOp9^online_cnn_1/depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2z
;online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp;online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp2~
=online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_1=online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_12~
=online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_2=online_cnn_1/batch_normalization_2/batchnorm/ReadVariableOp_22�
?online_cnn_1/batch_normalization_2/batchnorm/mul/ReadVariableOp?online_cnn_1/batch_normalization_2/batchnorm/mul/ReadVariableOp2Z
+online_cnn_1/conv2d_4/Conv2D/ReadVariableOp+online_cnn_1/conv2d_4/Conv2D/ReadVariableOp2Z
+online_cnn_1/conv2d_5/Conv2D/ReadVariableOp+online_cnn_1/conv2d_5/Conv2D/ReadVariableOp2t
8online_cnn_1/depthwise_conv2d_4/depthwise/ReadVariableOp8online_cnn_1/depthwise_conv2d_4/depthwise/ReadVariableOp2t
8online_cnn_1/depthwise_conv2d_5/depthwise/ReadVariableOp8online_cnn_1/depthwise_conv2d_5/depthwise/ReadVariableOp:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�	
�
,__inference_online_cnn_1_layer_call_fn_39546
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
:#@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39527j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:#@`
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
�	
�
,__inference_online_cnn_1_layer_call_fn_39731
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
:#@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39691j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:#@`
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
�!
�
__inference__traced_save_40248
file_prefixO
Ksavev2_online_cnn_1_depthwise_conv2d_4_depthwise_kernel_read_readvariableop;
7savev2_online_cnn_1_conv2d_4_kernel_read_readvariableopO
Ksavev2_online_cnn_1_depthwise_conv2d_5_depthwise_kernel_read_readvariableop;
7savev2_online_cnn_1_conv2d_5_kernel_read_readvariableopG
Csavev2_online_cnn_1_batch_normalization_2_gamma_read_readvariableopF
Bsavev2_online_cnn_1_batch_normalization_2_beta_read_readvariableopM
Isavev2_online_cnn_1_batch_normalization_2_moving_mean_read_readvariableopQ
Msavev2_online_cnn_1_batch_normalization_2_moving_variance_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Ksavev2_online_cnn_1_depthwise_conv2d_4_depthwise_kernel_read_readvariableop7savev2_online_cnn_1_conv2d_4_kernel_read_readvariableopKsavev2_online_cnn_1_depthwise_conv2d_5_depthwise_kernel_read_readvariableop7savev2_online_cnn_1_conv2d_5_kernel_read_readvariableopCsavev2_online_cnn_1_batch_normalization_2_gamma_read_readvariableopBsavev2_online_cnn_1_batch_normalization_2_beta_read_readvariableopIsavev2_online_cnn_1_batch_normalization_2_moving_mean_read_readvariableopMsavev2_online_cnn_1_batch_normalization_2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
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
�%
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_39420

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
�:
�
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39881
xN
4depthwise_conv2d_4_depthwise_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource: N
4depthwise_conv2d_5_depthwise_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource: @E
7batch_normalization_2_batchnorm_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_2_batchnorm_readvariableop_1_resource:@G
9batch_normalization_2_batchnorm_readvariableop_2_resource:@
identity��.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�+depthwise_conv2d_4/depthwise/ReadVariableOp�+depthwise_conv2d_5/depthwise/ReadVariableOp�
+depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_4_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0{
"depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            {
*depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_4/depthwiseDepthwiseConv2dNativex3depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_4/Conv2DConv2D%depthwise_conv2d_4/depthwise:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
h
re_lu_4/ReluReluconv2d_4/Conv2D:output:0*
T0*/
_output_shapes
:��������� t
dropout_4/IdentityIdentityre_lu_4/Relu:activations:0*
T0*/
_output_shapes
:��������� �
+depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_5_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0{
"depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             {
*depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_5/depthwiseDepthwiseConv2dNativedropout_4/Identity:output:03depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_5/Conv2DConv2D%depthwise_conv2d_5/depthwise:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
h
re_lu_5/ReluReluconv2d_5/Conv2D:output:0*
T0*/
_output_shapes
:���������@�
1global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
global_average_pooling2d_2/MeanMeanre_lu_5/Relu:activations:0:global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_2/batchnorm/mul_1Mul(global_average_pooling2d_2/Mean:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@{
dropout_5/IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   #   @   t
ReshapeReshapedropout_5/Identity:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:#@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:#@�
NoOpNoOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp,^depthwise_conv2d_4/depthwise/ReadVariableOp,^depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2Z
+depthwise_conv2d_4/depthwise/ReadVariableOp+depthwise_conv2d_4/depthwise/ReadVariableOp2Z
+depthwise_conv2d_5/depthwise/ReadVariableOp+depthwise_conv2d_5/depthwise/ReadVariableOp:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
(__inference_conv2d_5_layer_call_fn_40066

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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_39496w
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
�
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_40083

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
#__inference_signature_wrapper_39976
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
:#@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_39336j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:#@`
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
�
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_39505

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
C
'__inference_re_lu_4_layer_call_fn_40011

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
B__inference_re_lu_4_layer_call_and_return_conditional_losses_39467h
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
�	
�
,__inference_online_cnn_1_layer_call_fn_39837
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
:#@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39691j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:#@`
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
�
�
5__inference_batch_normalization_2_layer_call_fn_40107

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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_39373o
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
�
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_39522

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
�
q
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_39346

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
2__inference_depthwise_conv2d_4_layer_call_fn_39983

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
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39447w
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
�

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_39611

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
�c
�
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39953
xN
4depthwise_conv2d_4_depthwise_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource: N
4depthwise_conv2d_5_depthwise_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource: @K
=batch_normalization_2_assignmovingavg_readvariableop_resource:@M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@
identity��%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�+depthwise_conv2d_4/depthwise/ReadVariableOp�+depthwise_conv2d_5/depthwise/ReadVariableOp�
+depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_4_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0{
"depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            {
*depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_4/depthwiseDepthwiseConv2dNativex3depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_4/Conv2DConv2D%depthwise_conv2d_4/depthwise:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
h
re_lu_4/ReluReluconv2d_4/Conv2D:output:0*
T0*/
_output_shapes
:��������� \
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_4/dropout/MulMulre_lu_4/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:��������� a
dropout_4/dropout/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
ף;�
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� �
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� �
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:��������� �
+depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOp4depthwise_conv2d_5_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0{
"depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             {
*depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_5/depthwiseDepthwiseConv2dNativedropout_4/dropout/Mul_1:z:03depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_5/Conv2DConv2D%depthwise_conv2d_5/depthwise:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
h
re_lu_5/ReluReluconv2d_5/Conv2D:output:0*
T0*/
_output_shapes
:���������@�
1global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
global_average_pooling2d_2/MeanMeanre_lu_5/Relu:activations:0:global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������@~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_2/moments/meanMean(global_average_pooling2d_2/Mean:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:@�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference(global_average_pooling2d_2/Mean:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_2/batchnorm/mul_1Mul(global_average_pooling2d_2/Mean:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_5/dropout/MulMul)batch_normalization_2/batchnorm/add_1:z:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:���������@p
dropout_5/dropout/ShapeShape)batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   #   @   t
ReshapeReshapedropout_5/dropout/Mul_1:z:0Reshape/shape:output:0*
T0*"
_output_shapes
:#@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:#@�
NoOpNoOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp,^depthwise_conv2d_4/depthwise/ReadVariableOp,^depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2Z
+depthwise_conv2d_4/depthwise/ReadVariableOp+depthwise_conv2d_4/depthwise/ReadVariableOp2Z
+depthwise_conv2d_5/depthwise/ReadVariableOp+depthwise_conv2d_5/depthwise/ReadVariableOp:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
(__inference_conv2d_4_layer_call_fn_39999

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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_39458w
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
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40140

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
�
V
:__inference_global_average_pooling2d_2_layer_call_fn_40088

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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_39346i
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
�
q
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_40094

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
�
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_40031

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
�
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_39485

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
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_40201

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
C
'__inference_re_lu_5_layer_call_fn_40078

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
B__inference_re_lu_5_layer_call_and_return_conditional_losses_39505h
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
�	
�
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_40059

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
�
�
5__inference_batch_normalization_2_layer_call_fn_40120

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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_39420o
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
�(
�
!__inference__traced_restore_40282
file_prefix[
Aassignvariableop_online_cnn_1_depthwise_conv2d_4_depthwise_kernel:I
/assignvariableop_1_online_cnn_1_conv2d_4_kernel: ]
Cassignvariableop_2_online_cnn_1_depthwise_conv2d_5_depthwise_kernel: I
/assignvariableop_3_online_cnn_1_conv2d_5_kernel: @I
;assignvariableop_4_online_cnn_1_batch_normalization_2_gamma:@H
:assignvariableop_5_online_cnn_1_batch_normalization_2_beta:@O
Aassignvariableop_6_online_cnn_1_batch_normalization_2_moving_mean:@S
Eassignvariableop_7_online_cnn_1_batch_normalization_2_moving_variance:@

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
AssignVariableOpAssignVariableOpAassignvariableop_online_cnn_1_depthwise_conv2d_4_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp/assignvariableop_1_online_cnn_1_conv2d_4_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpCassignvariableop_2_online_cnn_1_depthwise_conv2d_5_depthwise_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_online_cnn_1_conv2d_5_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp;assignvariableop_4_online_cnn_1_batch_normalization_2_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp:assignvariableop_5_online_cnn_1_batch_normalization_2_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpAassignvariableop_6_online_cnn_1_batch_normalization_2_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpEassignvariableop_7_online_cnn_1_batch_normalization_2_moving_varianceIdentity_7:output:0"/device:CPU:0*
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
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_39458

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
�
E
)__inference_dropout_4_layer_call_fn_40021

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
D__inference_dropout_4_layer_call_and_return_conditional_losses_39474h
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
�-
�
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39763
input_12
depthwise_conv2d_4_39734:(
conv2d_4_39737: 2
depthwise_conv2d_5_39742: (
conv2d_5_39745: @)
batch_normalization_2_39750:@)
batch_normalization_2_39752:@)
batch_normalization_2_39754:@)
batch_normalization_2_39756:@
identity��-batch_normalization_2/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�*depthwise_conv2d_4/StatefulPartitionedCall�*depthwise_conv2d_5/StatefulPartitionedCall�
*depthwise_conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_1depthwise_conv2d_4_39734*
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
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39447�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_4/StatefulPartitionedCall:output:0conv2d_4_39737*
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_39458�
re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
B__inference_re_lu_4_layer_call_and_return_conditional_losses_39467�
dropout_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
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
D__inference_dropout_4_layer_call_and_return_conditional_losses_39474�
*depthwise_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0depthwise_conv2d_5_39742*
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
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_39485�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_5/StatefulPartitionedCall:output:0conv2d_5_39745*
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_39496�
re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
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
B__inference_re_lu_5_layer_call_and_return_conditional_losses_39505�
*global_average_pooling2d_2/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_39346�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0batch_normalization_2_39750batch_normalization_2_39752batch_normalization_2_39754batch_normalization_2_39756*
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_39373�
dropout_5/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_39522b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   #   @   {
ReshapeReshape"dropout_5/PartitionedCall:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:#@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:#@�
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall+^depthwise_conv2d_4/StatefulPartitionedCall+^depthwise_conv2d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2X
*depthwise_conv2d_4/StatefulPartitionedCall*depthwise_conv2d_4/StatefulPartitionedCall2X
*depthwise_conv2d_5/StatefulPartitionedCall*depthwise_conv2d_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
�
2__inference_depthwise_conv2d_5_layer_call_fn_40050

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
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_39485w
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
�%
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40174

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
�
�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_39496

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
�0
�
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39691
x2
depthwise_conv2d_4_39662:(
conv2d_4_39665: 2
depthwise_conv2d_5_39670: (
conv2d_5_39673: @)
batch_normalization_2_39678:@)
batch_normalization_2_39680:@)
batch_normalization_2_39682:@)
batch_normalization_2_39684:@
identity��-batch_normalization_2/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�*depthwise_conv2d_4/StatefulPartitionedCall�*depthwise_conv2d_5/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�
*depthwise_conv2d_4/StatefulPartitionedCallStatefulPartitionedCallxdepthwise_conv2d_4_39662*
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
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39447�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_4/StatefulPartitionedCall:output:0conv2d_4_39665*
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_39458�
re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
B__inference_re_lu_4_layer_call_and_return_conditional_losses_39467�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0*
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
D__inference_dropout_4_layer_call_and_return_conditional_losses_39611�
*depthwise_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0depthwise_conv2d_5_39670*
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
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_39485�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_5/StatefulPartitionedCall:output:0conv2d_5_39673*
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_39496�
re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
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
B__inference_re_lu_5_layer_call_and_return_conditional_losses_39505�
*global_average_pooling2d_2/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_39346�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0batch_normalization_2_39678batch_normalization_2_39680batch_normalization_2_39682batch_normalization_2_39684*
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_39420�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_39566b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   #   @   �
ReshapeReshape*dropout_5/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:#@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:#@�
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall+^depthwise_conv2d_4/StatefulPartitionedCall+^depthwise_conv2d_5/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2X
*depthwise_conv2d_4/StatefulPartitionedCall*depthwise_conv2d_4/StatefulPartitionedCall2X
*depthwise_conv2d_5/StatefulPartitionedCall*depthwise_conv2d_5/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_40073

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
�	
�
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39992

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
�
b
)__inference_dropout_4_layer_call_fn_40026

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
D__inference_dropout_4_layer_call_and_return_conditional_losses_39611w
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
�
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_39474

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
�0
�
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39795
input_12
depthwise_conv2d_4_39766:(
conv2d_4_39769: 2
depthwise_conv2d_5_39774: (
conv2d_5_39777: @)
batch_normalization_2_39782:@)
batch_normalization_2_39784:@)
batch_normalization_2_39786:@)
batch_normalization_2_39788:@
identity��-batch_normalization_2/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�*depthwise_conv2d_4/StatefulPartitionedCall�*depthwise_conv2d_5/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�
*depthwise_conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_1depthwise_conv2d_4_39766*
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
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39447�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_4/StatefulPartitionedCall:output:0conv2d_4_39769*
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_39458�
re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
B__inference_re_lu_4_layer_call_and_return_conditional_losses_39467�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0*
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
D__inference_dropout_4_layer_call_and_return_conditional_losses_39611�
*depthwise_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0depthwise_conv2d_5_39774*
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
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_39485�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_5/StatefulPartitionedCall:output:0conv2d_5_39777*
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_39496�
re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
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
B__inference_re_lu_5_layer_call_and_return_conditional_losses_39505�
*global_average_pooling2d_2/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_39346�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0batch_normalization_2_39782batch_normalization_2_39784batch_normalization_2_39786batch_normalization_2_39788*
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_39420�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_39566b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   #   @   �
ReshapeReshape*dropout_5/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:#@Z
IdentityIdentityReshape:output:0^NoOp*
T0*"
_output_shapes
:#@�
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall+^depthwise_conv2d_4/StatefulPartitionedCall+^depthwise_conv2d_5/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������  : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2X
*depthwise_conv2d_4/StatefulPartitionedCall*depthwise_conv2d_4/StatefulPartitionedCall2X
*depthwise_conv2d_5/StatefulPartitionedCall*depthwise_conv2d_5/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
b
)__inference_dropout_5_layer_call_fn_40184

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
D__inference_dropout_5_layer_call_and_return_conditional_losses_39566o
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
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_39373

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
StatefulPartitionedCall:0#@tensorflow/serving/predict:ë
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
,__inference_online_cnn_1_layer_call_fn_39546
,__inference_online_cnn_1_layer_call_fn_39816
,__inference_online_cnn_1_layer_call_fn_39837
,__inference_online_cnn_1_layer_call_fn_39731�
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
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39881
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39953
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39763
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39795�
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
 __inference__wrapped_model_39336input_1"�
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
J:H20online_cnn_1/depthwise_conv2d_4/depthwise_kernel
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
2__inference_depthwise_conv2d_4_layer_call_fn_39983�
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
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39992�
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
6:4 2online_cnn_1/conv2d_4/kernel
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
(__inference_conv2d_4_layer_call_fn_39999�
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40006�
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
'__inference_re_lu_4_layer_call_fn_40011�
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
B__inference_re_lu_4_layer_call_and_return_conditional_losses_40016�
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
)__inference_dropout_4_layer_call_fn_40021
)__inference_dropout_4_layer_call_fn_40026�
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
D__inference_dropout_4_layer_call_and_return_conditional_losses_40031
D__inference_dropout_4_layer_call_and_return_conditional_losses_40043�
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
J:H 20online_cnn_1/depthwise_conv2d_5/depthwise_kernel
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
2__inference_depthwise_conv2d_5_layer_call_fn_40050�
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
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_40059�
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
6:4 @2online_cnn_1/conv2d_5/kernel
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
(__inference_conv2d_5_layer_call_fn_40066�
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_40073�
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
'__inference_re_lu_5_layer_call_fn_40078�
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
B__inference_re_lu_5_layer_call_and_return_conditional_losses_40083�
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
:__inference_global_average_pooling2d_2_layer_call_fn_40088�
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
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_40094�
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
6:4@2(online_cnn_1/batch_normalization_2/gamma
5:3@2'online_cnn_1/batch_normalization_2/beta
>:<@ (2.online_cnn_1/batch_normalization_2/moving_mean
B:@@ (22online_cnn_1/batch_normalization_2/moving_variance
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
5__inference_batch_normalization_2_layer_call_fn_40107
5__inference_batch_normalization_2_layer_call_fn_40120�
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40140
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40174�
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
)__inference_dropout_5_layer_call_fn_40179
)__inference_dropout_5_layer_call_fn_40184�
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_40189
D__inference_dropout_5_layer_call_and_return_conditional_losses_40201�
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
#__inference_signature_wrapper_39976input_1"�
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
 __inference__wrapped_model_39336t/6NKML8�5
.�+
)�&
input_1���������  
� ".�+
)
output_1�
output_1#@�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40140bNKML3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40174bMNKL3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
5__inference_batch_normalization_2_layer_call_fn_40107UNKML3�0
)�&
 �
inputs���������@
p 
� "����������@�
5__inference_batch_normalization_2_layer_call_fn_40120UMNKL3�0
)�&
 �
inputs���������@
p
� "����������@�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40006k7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
(__inference_conv2d_4_layer_call_fn_39999^7�4
-�*
(�%
inputs���������
� " ���������� �
C__inference_conv2d_5_layer_call_and_return_conditional_losses_40073k67�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
(__inference_conv2d_5_layer_call_fn_40066^67�4
-�*
(�%
inputs��������� 
� " ����������@�
M__inference_depthwise_conv2d_4_layer_call_and_return_conditional_losses_39992k7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������
� �
2__inference_depthwise_conv2d_4_layer_call_fn_39983^7�4
-�*
(�%
inputs���������  
� " �����������
M__inference_depthwise_conv2d_5_layer_call_and_return_conditional_losses_40059k/7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0��������� 
� �
2__inference_depthwise_conv2d_5_layer_call_fn_40050^/7�4
-�*
(�%
inputs��������� 
� " ���������� �
D__inference_dropout_4_layer_call_and_return_conditional_losses_40031l;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
D__inference_dropout_4_layer_call_and_return_conditional_losses_40043l;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
)__inference_dropout_4_layer_call_fn_40021_;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
)__inference_dropout_4_layer_call_fn_40026_;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
D__inference_dropout_5_layer_call_and_return_conditional_losses_40189\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
D__inference_dropout_5_layer_call_and_return_conditional_losses_40201\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� |
)__inference_dropout_5_layer_call_fn_40179O3�0
)�&
 �
inputs���������@
p 
� "����������@|
)__inference_dropout_5_layer_call_fn_40184O3�0
)�&
 �
inputs���������@
p
� "����������@�
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_40094�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0������������������
� �
:__inference_global_average_pooling2d_2_layer_call_fn_40088wR�O
H�E
C�@
inputs4������������������������������������
� "!��������������������
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39763j/6NKML<�9
2�/
)�&
input_1���������  
p 
� " �
�
0#@
� �
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39795j/6MNKL<�9
2�/
)�&
input_1���������  
p
� " �
�
0#@
� �
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39881d/6NKML6�3
,�)
#� 
x���������  
p 
� " �
�
0#@
� �
G__inference_online_cnn_1_layer_call_and_return_conditional_losses_39953d/6MNKL6�3
,�)
#� 
x���������  
p
� " �
�
0#@
� �
,__inference_online_cnn_1_layer_call_fn_39546]/6NKML<�9
2�/
)�&
input_1���������  
p 
� "�#@�
,__inference_online_cnn_1_layer_call_fn_39731]/6MNKL<�9
2�/
)�&
input_1���������  
p
� "�#@�
,__inference_online_cnn_1_layer_call_fn_39816W/6NKML6�3
,�)
#� 
x���������  
p 
� "�#@�
,__inference_online_cnn_1_layer_call_fn_39837W/6MNKL6�3
,�)
#� 
x���������  
p
� "�#@�
B__inference_re_lu_4_layer_call_and_return_conditional_losses_40016h7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0��������� 
� �
'__inference_re_lu_4_layer_call_fn_40011[7�4
-�*
(�%
inputs��������� 
� " ���������� �
B__inference_re_lu_5_layer_call_and_return_conditional_losses_40083h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
'__inference_re_lu_5_layer_call_fn_40078[7�4
-�*
(�%
inputs���������@
� " ����������@�
#__inference_signature_wrapper_39976/6NKMLC�@
� 
9�6
4
input_1)�&
input_1���������  ".�+
)
output_1�
output_1#@
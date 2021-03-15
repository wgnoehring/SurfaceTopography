Search.setIndex({docnames:["contributing","development","index","installation","source/SurfaceTopography","source/SurfaceTopography.IO","source/SurfaceTopography.Nonuniform","source/SurfaceTopography.Tools","source/SurfaceTopography.Uniform","source/modules","testing","usage"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["contributing.rst","development.rst","index.rst","installation.rst","source/SurfaceTopography.rst","source/SurfaceTopography.IO.rst","source/SurfaceTopography.Nonuniform.rst","source/SurfaceTopography.Tools.rst","source/SurfaceTopography.Uniform.rst","source/modules.rst","testing.rst","usage.rst"],objects:{"":{SurfaceTopography:[4,0,0,"-"]},"SurfaceTopography.Converters":{UniformlyInterpolatedLineScan:[4,1,1,""],WrapAsNonuniformLineScan:[4,1,1,""]},"SurfaceTopography.Converters.UniformlyInterpolatedLineScan":{area_per_pt:[4,2,1,""],dim:[4,2,1,""],has_undefined_data:[4,2,1,""],heights:[4,2,1,""],is_periodic:[4,2,1,""],is_uniform:[4,2,1,""],nb_grid_pts:[4,2,1,""],physical_sizes:[4,2,1,""],pixel_size:[4,2,1,""],positions:[4,2,1,""]},"SurfaceTopography.Converters.WrapAsNonuniformLineScan":{dim:[4,2,1,""],heights:[4,2,1,""],is_periodic:[4,2,1,""],nb_grid_pts:[4,2,1,""],physical_sizes:[4,2,1,""],positions:[4,2,1,""],x_range:[4,2,1,""]},"SurfaceTopography.FFTTricks":{get_window_2D:[4,3,1,""],make_fft:[4,3,1,""]},"SurfaceTopography.Generation":{CapillaryWavesExact:[4,1,1,""],fourier_synthesis:[4,3,1,""],self_affine_prefactor:[4,3,1,""]},"SurfaceTopography.Generation.CapillaryWavesExact":{Error:[4,4,1,""],abs_q:[4,2,1,""],amplitude_distribution:[4,2,1,""],generate_amplitudes:[4,2,1,""],generate_phases:[4,2,1,""],get_negative_frequency_iterator:[4,2,1,""],get_topography:[4,2,1,""]},"SurfaceTopography.HeightContainer":{AbstractHeightContainer:[4,1,1,""],DecoratedTopography:[4,1,1,""],NonuniformLineScanInterface:[4,1,1,""],TopographyInterface:[4,1,1,""],UniformTopographyInterface:[4,1,1,""]},"SurfaceTopography.HeightContainer.AbstractHeightContainer":{Error:[4,5,1,""],apply:[4,2,1,""],communicator:[4,2,1,""],dim:[4,2,1,""],info:[4,2,1,""],is_periodic:[4,2,1,""],physical_sizes:[4,2,1,""],pipeline:[4,2,1,""]},"SurfaceTopography.HeightContainer.DecoratedTopography":{info:[4,2,1,""],pipeline:[4,2,1,""]},"SurfaceTopography.HeightContainer.NonuniformLineScanInterface":{heights:[4,2,1,""],is_MPI:[4,2,1,""],is_uniform:[4,2,1,""],nb_grid_pts:[4,2,1,""],positions:[4,2,1,""],positions_and_heights:[4,2,1,""],x_range:[4,2,1,""]},"SurfaceTopography.HeightContainer.TopographyInterface":{register_function:[4,2,1,""]},"SurfaceTopography.HeightContainer.UniformTopographyInterface":{area_per_pt:[4,2,1,""],communicator:[4,2,1,""],has_undefined_data:[4,2,1,""],heights:[4,2,1,""],is_domain_decomposed:[4,2,1,""],is_uniform:[4,2,1,""],nb_grid_pts:[4,2,1,""],pixel_size:[4,2,1,""],positions:[4,2,1,""],positions_and_heights:[4,2,1,""]},"SurfaceTopography.IO":{DI:[5,0,0,"-"],FromFile:[5,0,0,"-"],H5:[5,0,0,"-"],IBW:[5,0,0,"-"],MI:[5,0,0,"-"],Matlab:[5,0,0,"-"],NC:[5,0,0,"-"],NPY:[5,0,0,"-"],OPDx:[5,0,0,"-"],Reader:[5,0,0,"-"],detect_format:[5,3,1,""],open_topography:[5,3,1,""],read_topography:[5,3,1,""]},"SurfaceTopography.IO.DI":{DIReader:[5,1,1,""]},"SurfaceTopography.IO.DI.DIReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.FromFile":{AscReader:[5,1,1,""],HGTReader:[5,1,1,""],MatrixReader:[5,1,1,""],OPDReader:[5,1,1,""],X3PReader:[5,1,1,""],XYZReader:[5,1,1,""],binary:[5,3,1,""],get_unit_conversion_factor:[5,3,1,""],is_binary_stream:[5,3,1,""],make_wrapped_reader:[5,3,1,""],mangle_height_unit:[5,3,1,""],mask_undefined:[5,3,1,""],read_asc:[5,3,1,""],read_hgt:[5,3,1,""],read_matrix:[5,3,1,""],read_opd:[5,3,1,""],read_x3p:[5,3,1,""],read_xyz:[5,3,1,""],text:[5,3,1,""]},"SurfaceTopography.IO.FromFile.AscReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.FromFile.HGTReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.FromFile.MatrixReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.FromFile.OPDReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.FromFile.X3PReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.FromFile.XYZReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.H5":{H5Reader:[5,1,1,""]},"SurfaceTopography.IO.H5.H5Reader":{channels:[5,2,1,""],close:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.IBW":{IBWReader:[5,1,1,""]},"SurfaceTopography.IO.IBW.IBWReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.MI":{Channel:[5,1,1,""],MIFile:[5,1,1,""],MIReader:[5,1,1,""],read_header_image:[5,3,1,""],read_header_spect:[5,3,1,""]},"SurfaceTopography.IO.MI.MIReader":{channels:[5,2,1,""],info:[5,2,1,""],process_header:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.Matlab":{MatReader:[5,1,1,""]},"SurfaceTopography.IO.Matlab.MatReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.NC":{NCReader:[5,1,1,""],write_nc:[5,3,1,""]},"SurfaceTopography.IO.NC.NCReader":{channels:[5,2,1,""],close:[5,2,1,""],communicator:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.NPY":{NPYReader:[5,1,1,""],save_npy:[5,3,1,""]},"SurfaceTopography.IO.NPY.NPYReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.OPDx":{DektakBuf:[5,1,1,""],DektakItem:[5,1,1,""],DektakItemData:[5,1,1,""],DektakMatrix:[5,1,1,""],DektakQuantUnit:[5,1,1,""],DektakRawPos1D:[5,1,1,""],DektakRawPos2D:[5,1,1,""],OPDxReader:[5,1,1,""],build_matrix:[5,3,1,""],create_meta:[5,3,1,""],find_1d_data:[5,3,1,""],find_2d_data:[5,3,1,""],find_2d_data_matrix:[5,3,1,""],read_dimension2d_content:[5,3,1,""],read_double:[5,3,1,""],read_float:[5,3,1,""],read_int16:[5,3,1,""],read_int32:[5,3,1,""],read_int64:[5,3,1,""],read_item:[5,3,1,""],read_name:[5,3,1,""],read_named_struct:[5,3,1,""],read_quantunit_content:[5,3,1,""],read_structured:[5,3,1,""],read_varlen:[5,3,1,""],read_with_check:[5,3,1,""],reformat_dict:[5,3,1,""]},"SurfaceTopography.IO.OPDx.OPDxReader":{channels:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.Reader":{CannotDetectFileFormat:[5,5,1,""],ChannelInfo:[5,1,1,""],CorruptFile:[5,5,1,""],FileFormatMismatch:[5,5,1,""],ReadFileError:[5,5,1,""],ReaderBase:[5,1,1,""],UnknownFileFormatGiven:[5,5,1,""]},"SurfaceTopography.IO.Reader.ChannelInfo":{area_per_pt:[5,2,1,""],dim:[5,2,1,""],index:[5,2,1,""],info:[5,2,1,""],is_periodic:[5,2,1,""],name:[5,2,1,""],nb_grid_pts:[5,2,1,""],physical_sizes:[5,2,1,""],pixel_size:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.IO.Reader.ReaderBase":{channels:[5,2,1,""],close:[5,2,1,""],default_channel:[5,2,1,""],description:[5,2,1,""],format:[5,2,1,""],name:[5,2,1,""],topography:[5,2,1,""]},"SurfaceTopography.Nonuniform":{Autocorrelation:[6,0,0,"-"],Detrending:[6,0,0,"-"],PowerSpectrum:[6,0,0,"-"],ScalarParameters:[6,0,0,"-"],VariableBandwidth:[6,0,0,"-"],common:[6,0,0,"-"]},"SurfaceTopography.Nonuniform.Autocorrelation":{height_difference_autocorrelation_1D:[6,3,1,""],height_height_autocorrelation_1D:[6,3,1,""]},"SurfaceTopography.Nonuniform.Detrending":{polyfit:[6,3,1,""]},"SurfaceTopography.Nonuniform.PowerSpectrum":{apply_window:[6,3,1,""],dsinc:[6,3,1,""],ft_one_sided_triangle:[6,3,1,""],ft_rectangle:[6,3,1,""],power_spectrum_1D:[6,3,1,""],sinc:[6,3,1,""]},"SurfaceTopography.Nonuniform.ScalarParameters":{rms_curvature:[6,3,1,""],rms_height:[6,3,1,""],rms_slope:[6,3,1,""]},"SurfaceTopography.Nonuniform.VariableBandwidth":{checkerboard_detrend:[6,3,1,""],variable_bandwidth:[6,3,1,""]},"SurfaceTopography.Nonuniform.common":{bandwidth:[6,3,1,""],derivative:[6,3,1,""]},"SurfaceTopography.NonuniformLineScan":{DecoratedNonuniformTopography:[4,1,1,""],DetrendedNonuniformTopography:[4,1,1,""],NonuniformLineScan:[4,1,1,""],ScaledNonuniformTopography:[4,1,1,""]},"SurfaceTopography.NonuniformLineScan.DecoratedNonuniformTopography":{dim:[4,2,1,""],is_periodic:[4,2,1,""],nb_grid_pts:[4,2,1,""],physical_sizes:[4,2,1,""],positions:[4,2,1,""],squeeze:[4,2,1,""],x_range:[4,2,1,""]},"SurfaceTopography.NonuniformLineScan.DetrendedNonuniformTopography":{coeffs:[4,2,1,""],curvatures:[4,2,1,""],detrend_mode:[4,2,1,""],heights:[4,2,1,""],is_periodic:[4,2,1,""],positions:[4,2,1,""],stringify_plane:[4,2,1,""],x_range:[4,2,1,""]},"SurfaceTopography.NonuniformLineScan.NonuniformLineScan":{dim:[4,2,1,""],heights:[4,2,1,""],is_periodic:[4,2,1,""],is_uniform:[4,2,1,""],nb_grid_pts:[4,2,1,""],physical_sizes:[4,2,1,""],positions:[4,2,1,""],x_range:[4,2,1,""]},"SurfaceTopography.NonuniformLineScan.ScaledNonuniformTopography":{heights:[4,2,1,""],scale_factor:[4,2,1,""]},"SurfaceTopography.Special":{PlasticTopography:[4,1,1,""],make_sphere:[4,3,1,""]},"SurfaceTopography.Special.PlasticTopography":{hardness:[4,2,1,""],heights:[4,2,1,""],name:[4,4,1,""],plastic_area:[4,2,1,""],plastic_displ:[4,2,1,""],undeformed_profile:[4,2,1,""]},"SurfaceTopography.Tools":{common:[7,0,0,"-"]},"SurfaceTopography.Tools.common":{compute_wavevectors:[7,3,1,""],fftn:[7,3,1,""],ifftn:[7,3,1,""]},"SurfaceTopography.Uniform":{Autocorrelation:[8,0,0,"-"],Detrending:[8,0,0,"-"],Filtering:[8,0,0,"-"],Interpolation:[8,0,0,"-"],PowerSpectrum:[8,0,0,"-"],ScalarParameters:[8,0,0,"-"],VariableBandwidth:[8,0,0,"-"],common:[8,0,0,"-"]},"SurfaceTopography.Uniform.Autocorrelation":{autocorrelation_1D:[8,3,1,""],autocorrelation_2D:[8,3,1,""]},"SurfaceTopography.Uniform.Detrending":{shift_and_tilt:[8,3,1,""],tilt_and_curvature:[8,3,1,""],tilt_from_height:[8,3,1,""]},"SurfaceTopography.Uniform.Filtering":{FilteredUniformTopography:[8,1,1,""],LongCutTopography:[8,1,1,""],ShortCutTopography:[8,1,1,""]},"SurfaceTopography.Uniform.Filtering.FilteredUniformTopography":{filter_function:[8,2,1,""],heights:[8,2,1,""],is_filter_isotropic:[8,2,1,""],name:[8,4,1,""]},"SurfaceTopography.Uniform.Filtering.LongCutTopography":{cutoff_wavelength:[8,2,1,""],cutoff_wavevector:[8,2,1,""],name:[8,4,1,""]},"SurfaceTopography.Uniform.Filtering.ShortCutTopography":{cutoff_wavelength:[8,2,1,""],cutoff_wavevector:[8,2,1,""],name:[8,4,1,""]},"SurfaceTopography.Uniform.Interpolation":{MirrorStichedTopography:[8,1,1,""],bicubic_interpolator:[8,3,1,""],interpolate_fourier:[8,3,1,""]},"SurfaceTopography.Uniform.Interpolation.MirrorStichedTopography":{heights:[8,2,1,""],is_periodic:[8,2,1,""],nb_grid_pts:[8,2,1,""],physical_sizes:[8,2,1,""],positions:[8,2,1,""]},"SurfaceTopography.Uniform.PowerSpectrum":{power_spectrum_1D:[8,3,1,""],power_spectrum_2D:[8,3,1,""]},"SurfaceTopography.Uniform.ScalarParameters":{rms_curvature:[8,3,1,""],rms_height:[8,3,1,""],rms_laplacian:[8,3,1,""],rms_slope:[8,3,1,""]},"SurfaceTopography.Uniform.VariableBandwidth":{checkerboard_detrend:[8,3,1,""],variable_bandwidth:[8,3,1,""]},"SurfaceTopography.Uniform.common":{FilledTopography:[8,1,1,""],bandwidth:[8,3,1,""],derivative:[8,3,1,""],domain_decompose:[8,3,1,""],fourier_derivative:[8,3,1,""],plot:[8,3,1,""]},"SurfaceTopography.Uniform.common.FilledTopography":{has_undefined_data:[8,2,1,""],heights:[8,2,1,""]},"SurfaceTopography.UniformLineScanAndTopography":{CompoundTopography:[4,1,1,""],DecoratedUniformTopography:[4,1,1,""],DetrendedUniformTopography:[4,1,1,""],ScaledUniformTopography:[4,1,1,""],Topography:[4,1,1,""],TranslatedTopography:[4,1,1,""],TransposedUniformTopography:[4,1,1,""],UniformLineScan:[4,1,1,""]},"SurfaceTopography.UniformLineScanAndTopography.CompoundTopography":{heights:[4,2,1,""],name:[4,4,1,""]},"SurfaceTopography.UniformLineScanAndTopography.DecoratedUniformTopography":{area_per_pt:[4,2,1,""],dim:[4,2,1,""],has_undefined_data:[4,2,1,""],is_domain_decomposed:[4,2,1,""],is_periodic:[4,2,1,""],nb_grid_pts:[4,2,1,""],nb_subdomain_grid_pts:[4,2,1,""],physical_sizes:[4,2,1,""],pixel_size:[4,2,1,""],positions:[4,2,1,""],positions_and_heights:[4,2,1,""],squeeze:[4,2,1,""],subdomain_locations:[4,2,1,""],subdomain_slices:[4,2,1,""]},"SurfaceTopography.UniformLineScanAndTopography.DetrendedUniformTopography":{coeffs:[4,2,1,""],curvatures:[4,2,1,""],detrend_mode:[4,2,1,""],heights:[4,2,1,""],is_periodic:[4,2,1,""],stringify_plane:[4,2,1,""]},"SurfaceTopography.UniformLineScanAndTopography.ScaledUniformTopography":{heights:[4,2,1,""],scale_factor:[4,2,1,""]},"SurfaceTopography.UniformLineScanAndTopography.Topography":{area_per_pt:[4,2,1,""],communicator:[4,2,1,""],dim:[4,2,1,""],has_undefined_data:[4,2,1,""],heights:[4,2,1,""],is_periodic:[4,2,1,""],is_uniform:[4,2,1,""],nb_grid_pts:[4,2,1,""],nb_subdomain_grid_pts:[4,2,1,""],physical_sizes:[4,2,1,""],pixel_size:[4,2,1,""],positions:[4,2,1,""],positions_and_heights:[4,2,1,""],save:[4,2,1,""],subdomain_locations:[4,2,1,""],subdomain_slices:[4,2,1,""]},"SurfaceTopography.UniformLineScanAndTopography.TranslatedTopography":{heights:[4,2,1,""],name:[4,4,1,""],offset:[4,2,1,""]},"SurfaceTopography.UniformLineScanAndTopography.TransposedUniformTopography":{heights:[4,2,1,""],nb_grid_pts:[4,2,1,""],physical_sizes:[4,2,1,""],positions:[4,2,1,""]},"SurfaceTopography.UniformLineScanAndTopography.UniformLineScan":{area_per_pt:[4,2,1,""],dim:[4,2,1,""],has_undefined_data:[4,2,1,""],heights:[4,2,1,""],is_domain_decomposed:[4,2,1,""],is_periodic:[4,2,1,""],is_uniform:[4,2,1,""],nb_grid_pts:[4,2,1,""],physical_sizes:[4,2,1,""],pixel_size:[4,2,1,""],positions:[4,2,1,""],save:[4,2,1,""]},"SurfaceTopography.common":{radial_average:[4,3,1,""]},SurfaceTopography:{Converters:[4,0,0,"-"],FFTTricks:[4,0,0,"-"],Generation:[4,0,0,"-"],HeightContainer:[4,0,0,"-"],IO:[5,0,0,"-"],Interpolation:[4,0,0,"-"],Nonuniform:[6,0,0,"-"],NonuniformLineScan:[4,0,0,"-"],Special:[4,0,0,"-"],Tools:[7,0,0,"-"],Uniform:[8,0,0,"-"],UniformLineScanAndTopography:[4,0,0,"-"],common:[4,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:exception"},terms:{"013001":[2,4],"04577566528320313":5,"0502419":4,"100":8,"10000":5,"1088":2,"111":8,"16bit":5,"18_line_scan":0,"2005":4,"2011":4,"2017":[2,4],"2018":0,"2051":2,"215004":4,"256":5,"29638271279074097":5,"2h_":6,"2h_i":6,"2x_":6,"2x_i":6,"32bit":5,"64bit":5,"672x":2,"9999":5,"999999999998":5,"\u0148":8,"\u0148_z":8,"\u03c6":4,"abstract":[4,5],"byte":5,"case":[0,4],"class":[0,2,3,4,5,6,7,8,11],"default":[0,4,5,6,8,11],"export":3,"float":[4,5,6,8],"function":[0,2,4,6,7,8],"import":[0,2,11],"int":[4,5,6,8],"long":[4,5],"m\u00fcser":11,"new":[0,5,6,11],"return":[4,5,6,7,8,11],"short":[4,5,6],"static":4,"true":[0,4,8],"try":3,"while":5,For:[0,4,5,6,8,10,11],HAS:5,Has:5,NOT:5,The:[0,3,4,5,6,8,11],Then:6,There:[4,11],These:[5,11],WITH:5,With:8,X_s:8,__init__:5,_build:0,_function:4,a_k:6,a_l:6,a_xi:8,aa51f8:2,abl:5,about:5,abov:6,abs:4,abs_q:4,absolut:[5,8],abspo:5,abstractheightcontain:4,access:[5,11],accord:[5,7],acf:[6,8],achiev:4,acquir:5,across:11,actual:[4,5],adapt:3,add:[0,8],added:0,addit:[4,5,11],address:0,adhes:[4,11],adjust:4,affect:5,affin:4,after:[0,4,11],again:0,against:11,algorithm:[4,6],alia:4,all:[0,3,4,5,6,8,11],allow:[5,6,8],almost:0,along:5,alreadi:3,also:[0,5,11],altern:11,alwai:[0,5,11],ambigu:8,amplitud:[4,8],amplitude_distribut:4,amplitudeerror:5,analysi:[2,6,8],analyz:[2,6,8],ani:[0,1,4,11],anoth:0,api:0,apidoc:0,append:[5,11],appendic:4,appli:[4,5,6,8],applic:[4,5,6,8],apply_window:6,appropri:4,approx:6,approxim:[4,6],area:[5,7],area_per_pt:[4,5],arg:[4,5,8],arg_min:8,argument:[4,5,7,8],around:4,arr:[7,8],arr_out:8,arrai:[4,5,6,7,8,11],array_lik:[4,6,8],artefact:5,arxiv:4,ascend:6,ascread:5,aspect:4,assert:8,assertionerror:8,assum:[6,8,11],aswel:5,attribut:5,author:2,autocorrel:[2,4,5,9],autocorrelation_1d:8,autocorrelation_2d:8,autom:10,automat:[0,3,6,8],auxiliari:4,avail:[3,5,11],averag:[4,8,11],awai:4,axes:[5,11],axi:5,bandwidth:[2,6,8],base:[0,4,5,8],basic:[2,11],becaus:[0,4,8],been:[5,6,11],befor:[0,4,6,8,11],begin:[4,8],behaviour:0,belong:0,below:[4,8,11],bending_stiff:4,between:[4,6,11],beyond:10,bicub:[3,8],bicubic_interpol:8,big:8,bin:[4,6,7,8],binari:5,block:0,bool:[4,5,8],both:11,bound:[6,8],boundari:[4,5,6,8],branch:2,brute:6,buf:5,buffer:5,bug:0,build:[0,1,3,10],build_matrix:5,c_0:4,c_all:8,c_r:4,c_xy:4,calcul:[5,11],call:[0,4,5,6],can:[0,2,3,4,5,6,7,8,10,11],cannot:5,cannotdetectfileformat:5,capillari:4,capillarywavesexact:4,care:0,carri:[6,8,11],cartesian:[8,11],cdot:4,center:4,centr:4,central:6,certain:11,challeng:11,chang:0,changelog:0,channel:[5,11],channel_index:5,channelinfo:5,check:5,checkerboard:[6,8],checkerboard_detrend:[6,8],child:4,choos:[5,8,10],circl:4,circular:8,cite:2,class_nam:5,classmethod:[4,5],cleaner:0,clone:3,close:5,closest:6,code:[1,2,11],coeff:[4,8],coeffici:6,colorbar:0,com:3,combin:4,comm:[0,5],comm_seri:0,comm_world:0,command:[3,11],commandlin:11,comminc:0,commit:2,common:9,commun:[0,4,5,8],compar:11,compat:[0,4,6],compil:[1,2,3],complement:4,complex:[7,11],compon:[7,8],compound_topographi:4,compoundtopographi:4,compress:4,comput:[4,5,6,7,8,10,11],computationalmechan:3,compute_wavevector:7,concaten:11,cond:4,conden:4,configur:0,conftest:0,conjug:11,connect:[6,11],consist:4,constrain:11,constraint:11,construct:[4,6],contact:[4,11],contain:[0,1,2,4,5,6,8,11],content:[0,9],contribut:[1,2,6],conveni:4,convent:[5,7],convers:[4,5],convert:9,coordin:[4,11],copyright:0,correct:[4,5,6,8],correspond:[4,6,8,11],corruptfil:5,cos:6,cosinu:8,cosinusoid:8,could:0,cpp:3,cppflag:3,creat:[0,4],create_meta:5,creation:0,current:[3,4,5,11],curvatur:[4,8,11],cutoff:[4,6,8],cutoff_wavelength:8,cutoff_wavevector:8,cython:[0,1],data:[2,4,5,6,8],date:4,datetim:4,deal:0,debug:10,decompos:8,decomposed_topographi:8,decomposit:[4,5,6,8],decoratednonuniformtopographi:4,decoratedtopographi:4,decorateduniformtopographi:[4,8],def:0,default_channel:[5,11],defin:[0,4,8,11],deform:[4,11],deg:6,degre:6,dektakbuf:5,dektakitem:5,dektakitemdata:5,dektakmatrix:5,dektakquantunit:5,dektakrawpos1d:5,dektakrawpos2d:5,delta:6,densiti:[4,6,8],depend:[3,5,7,11],deriv:[6,8],describ:[4,5,11],descript:[0,4,5],detail:7,detect:5,detect_format:5,determin:[4,5,6],detrend:[4,9,11],detrend_mod:[4,11],detrendednonuniformtopographi:4,detrendeduniformtopographi:4,develop:[2,10],dft:8,dict:5,dictionari:[4,5],differ:[5,6,8],dim:[4,5,8,11],dimens:[4,5,7,8,11],dimension:[4,6,7,8,11],diread:5,direct:[2,4,5,8,11],directli:[4,5],directoi:3,directori:[2,10],discard:8,disk:5,distanc:[4,6,8],distribut:4,distro:4,divid:5,divisor:5,doc:[0,5],document:[5,11],doe:[6,11],doesn:[0,5],dof:8,doi:2,domain:[4,5,6,7,8],domain_decompos:8,don:4,doubl:5,down:[6,8],dsinc:6,due:5,dure:[3,11],each:[5,6,8,11],easili:2,edg:[4,8],edu:8,effect:8,efficient:0,either:[5,11],elast:11,element:[5,6,8],elimin:8,els:4,email:0,emul:5,end:[4,8,11],enh:0,enhanc:0,entri:[4,5],env:[0,1],environ:[0,1,3],equal:[4,5,6],equat:[4,6,8],error:[4,5],etc:[4,5,7],even:[4,8],everytim:11,exactli:6,exampl:[0,4,5,8,11],except:[4,5],exclus:11,execut:[0,4,10,11],exist:[6,7],expand:6,expect:[3,11],experienc:3,experiment:2,explanatori:4,explict:0,expon:4,expos:11,express:4,extens:5,extra:5,extract:[4,5],factor:[4,5,6,11],fail:3,fals:[0,4,5,8],far:4,featur:[0,11],few:4,fft:[4,6,7,8],fftfreq:8,fftn:7,ffttrick:9,fftw3:3,fftwdir:3,field:[4,8],fig:0,file:[0,4,5,11],file_format_exampl:11,file_path:5,fileformatmismatch:5,fileionetcdf:3,filelik:5,filenam:5,fill:5,fill_valu:8,filledtopographi:8,filter:[2,4,9],filter_funct:8,filtered_topographi:8,filtereduniformtopographi:8,find:[3,11],find_1d_data:5,find_2d_data:5,find_2d_data_matrix:5,finer:[6,8],first:[5,6,8,11],fit:4,fix:0,fixm:5,fixtur:0,flag:[8,10],flat:[4,11],fluctuat:6,fmt:4,fname:4,fobj:5,folder:[0,7,11],follow:[0,3,4,5,6,8,10,11],forc:[0,4,6],format:[4,5],found:[0,4,5],fouri:4,fourier:[4,6,8],fourier_deriv:8,fourier_synthesi:4,frac:[4,6,8],fractal:4,free:8,frequenc:[4,8],frequenca:8,friction:4,from:[0,2,4,5,8,11],fromfil:[4,9],front:11,frozen:4,ft_one_sided_triangl:6,ft_rectangl:6,full:[4,5,6,8,11],full_output:8,fulli:11,func:5,funcion:0,gener:[0,5,9],generate_amplitud:4,generate_phas:4,geometr:4,geometri:4,get:[5,8,11],get_negative_frequency_iter:4,get_siz:0,get_topographi:4,get_unit_conversion_factor:5,get_window:8,get_window_2d:4,git:3,github:3,give:[4,5,6],given:[4,5,6,8,11],global:5,good:[0,11],gradient:[8,11],grid:[4,5,6,8,11],gwyddion:11,h5reader:5,h_0:8,h_i:[6,8],half:8,halfspac:4,handl:[2,5,8],hann:[4,6,8],happen:0,hard:[4,11],hard_wal:11,has:[4,5,6,11],has_undefined_data:[4,8],hash:5,hash_tabl:5,have:[0,3,4,5,6,8,10,11],header:5,height:[2,4,5,6,8,11],height_difference_autocorrelation_1d:6,height_height_autocorrelation_1d:6,height_scale_factor:5,heightcontain:[5,9],help:11,helper:[4,6,7,8],henc:6,here:[3,4,5,6,8,11],heurist:5,hex:5,hgtreader:5,hierarchi:4,ho_:4,hold:[5,8],home:3,homebrew:3,homogen:11,how:[6,11],html:[0,5],http:[2,3,4,5,8],hurst:4,ibw:[4,9],ibwread:5,idea:[0,8],ideal:4,ident:[5,6,8],identifi:5,ifft:7,ifftn:7,imag:[8,11],imaginari:8,immedi:4,impenetr:11,implement:[0,2,3,5,8,11],imshow:[5,11],imtol:8,includ:[0,3,5],incompat:[0,5],increasingli:[6,8],independ:5,index:[0,2,5,8],indic:[0,11],individu:[6,8],inf:[4,5,8],info:[4,5,8,11],inform:[0,4,5,6,8,11],infti:4,input:[5,6,7,8],inspect:5,instal:[0,1,2,11],instanc:[5,8],instanti:4,instead:[0,5],int_0:6,int_:[4,6],integ:5,integr:[6,7],inter:5,interact:[4,11],interfac:[4,5,11],interol:8,interpol:[6,9,11],interpolate_fouri:8,interpret:[0,1,5,8,11],intracomm:[4,5],invers:4,invert:5,invok:3,ipython:11,is_binary_stream:5,is_domain_decompos:4,is_filter_isotrop:8,is_mpi:4,is_period:[4,5,8],is_uniform:4,is_unit:5,iso:4,isotrop:8,item:5,its:[0,5],itself:[4,5],jacob:[2,4],jung:[2,4],jupyt:11,just:[0,4,11],keep:5,keer:11,kei:[4,5],keyword:[4,5,7],kfn:4,kind:[4,6,8],know:11,kwarg:[4,5],label:[0,5,11],lambda:[0,4,6,8],lambda_max:4,lambda_min:4,lapack:2,laplacian:8,larg:4,large_wavelength_cutoff:0,later:[4,8,11],latter:0,layout:4,ldflag:3,lead:[3,11],left:[4,5,6,8,11],length:[4,5,6,7,8],less:8,let:0,lib:[3,5],librari:3,like:11,line:[0,3,4,5,6,8,11],line_scan:6,linear:[6,11],link:3,list:[4,5,6,11],live:[4,11],load:[5,11],local:[3,8],locat:[5,8],location_matrix:8,log:[0,6],long_cutoff:4,longcut_filtered_topographi:8,longcuttopographi:8,longest:4,lower:[5,6,8,11],lower_bound:[6,8],maco:3,made:[5,8],magnif:[6,8],mai:[3,5,11],main:[6,8,10],maint:0,mainten:0,make:[0,1,4,5,8],make_fft:4,make_spher:4,make_wrapped_read:5,mangle_height_unit:5,map:[4,5,6,8,11],mark:[0,11],markdown:5,martin:11,mask:[4,5,8],mask_funct:8,mask_undefin:5,mass_dens:4,mat:4,match:5,math:[4,6,8],matlab:[4,9],matplotlib:[0,5,8,11],matread:5,matrix:5,matrixread:5,matter:4,max:6,maximum:[4,6,8],maxval:5,mayb:4,mean:[4,6,8,11],measur:[4,5,11],mechan:[4,11],member:4,memori:[4,5],messag:5,meta:[5,11],metadata:[5,11],method:[0,3,5,11],metrol:[2,4],mifil:5,might:4,min:6,min_:6,minim:[4,6,8],minimum:[4,6,8],minor:0,miread:5,mirror:[5,8],mirrorstichedtopographi:8,miss:3,mistak:5,mit:8,mode:4,modifi:[0,1,11],modul:[2,9,11],more:[5,8,11],mpi4pi:[4,5,8],mpi:[4,5,8],mpirun:10,mpistub:[4,5],mufft:[3,4,8],multidimension:8,multipl:[4,5],multipli:[4,5,8],muspectr:3,must:[0,5],my_surfac:11,n_x:8,n_y:8,n_z:8,name:[0,4,5,6,8,11],nan:5,natur:4,navig:0,nb_dim:7,nb_grid_pt:[4,5,6,7,8,11],nb_grid_pts_cutoff:[6,8],nb_point:4,nb_subdomain_grid_pt:[4,5,8],nbin:[4,8],nbyte:5,ncreader:5,ndarrai:4,necessari:5,need:[0,1,3,4,5,6,11],netcdf3_64bit_offset:5,netcdf:[3,5],netcdfdir:3,newest:3,next:5,nicer:5,ninterpol:6,niquist:8,non:[4,8],none:[4,5,6,8],nonoverlap:[6,8],nonperiod:[6,8],nonuniform:[2,4,8,9,11],nonuniformlinescan:[6,9],nonuniformlinescaninterfac:4,nonuniformtopographi:11,normal:[4,8],normalize_window:8,note:[0,1,2,3,4,5,7,8],now:[0,5,8],npy:[4,9],npyread:5,number:[0,4,5,6,8,10,11],numer:4,numpi:[3,4,5,8],numpydoc:0,numpysurfac:4,numpytxttopographi:4,nyquist:8,obj:[5,6],object:[4,5,6,8],obtain:[4,5,11],obviou:5,offer:5,offset:4,often:[0,3,5],older:0,omit:6,onc:0,one:[0,3,4,5,6,7,8,10,11],onli:[0,4,5,6,8,11],onto:4,opd:11,opdread:5,opdx:[4,9],opdxread:5,open:5,open_topographi:[5,11],openbla:2,opengp:5,oper:[6,8,11],opt:3,option:[3,4,5,6,8,10,11],order:[0,4,5,6,8],org:[2,4,5],origin:[5,8,11],oscil:8,other:[3,4,5,10,11],otherwis:[4,8],out:[5,6,8,11],output:[5,8,11],outsid:4,over:[5,8],overrid:[5,8],oversubscrib:10,overwrit:11,overwritten:5,own:[0,4,11],packag:[2,9],pad:[4,8],page:2,pai:4,paraboloid:4,parallel:[0,3,5],paralleliz:10,param:5,paramet:[0,1,3,4,5,6,8],parameter:[6,8],parent:4,parent_topographi:8,part:[5,8],parti:5,partial:[6,8],particular:3,pass:[0,4,5,6],pastewka:[2,4],path:[0,1,3,5],pcolormesh:[0,5,11],pdb:10,pdf:8,pep:0,per:[4,5,7,8,10],perform:[4,6,8],period:[4,5,6,8],periodicfftelastichalfspac:0,perpendicular:11,persson:4,phase:[4,8],phy:4,phyiscal:5,physic:[4,5,8],physical_s:[4,5,6,7,8,11],physical_si:5,piec:6,pip:2,pipelin:4,pixel:[4,5],pixel_s:[4,5],pkg_config_path:3,pkgconfig:3,plan:1,plane:[4,6,8],plastic:4,plastic_area:4,plastic_displ:4,plastic_topographi:4,plastictopographi:4,pleas:[0,1,2],plot:[5,8,11],plt:[0,5,11],pnetcdfdir:3,point:[4,5,6,8,11],polonski:11,polyfit:6,polynomi:[4,6],portion:[0,1],pos:5,posit:[4,5,6,8,11],positions_and_height:[4,5,11],possibl:11,power:[2,4,6,8,11],power_spectrum:7,power_spectrum_1d:[6,8,11],power_spectrum_2d:[8,11],powerspectrum:[4,9],prefactor:4,prepend:0,prescrib:5,present:[4,5,6,8,11],pressur:0,previou:0,previous:5,price:4,primari:5,primary_channel_nam:5,prime_:6,print:[0,11],probabl:[3,5],problem:[2,6],process:[4,5,8,11],process_head:5,processor:[0,5,10],product:5,profil:[4,5],program:5,progress:4,progress_callback:4,prop:[2,4],properti:[4,5,8,11],provid:[3,11],psd:[4,5,6,11],pseudo:11,pull:3,purpos:[0,1,2,10],put:6,pyco:[0,1,4,5,11],pyplot:[0,11],pytest:[0,10],pytestmark:0,python3:[0,1,3],python:[0,1,2,3,10,11],pyx:[0,1],q_x:8,q_y:8,quadrant:4,quantiti:[4,5],quantunit:5,quarter:4,question:11,r62:4,r_averag:4,r_edg:4,rac:4,radial:[4,8,11],radial_averag:4,radiu:4,rais:[4,5,8],ramisetti:4,random:4,randomli:4,rang:4,rather:[0,11],ratio:4,raw:[5,11],read:[1,2,5,11],read_asc:5,read_dimension2d_cont:5,read_doubl:5,read_float:5,read_header_imag:5,read_header_spect:5,read_hgt:5,read_int16:5,read_int32:5,read_int64:5,read_item:5,read_matrix:5,read_nam:5,read_named_struct:5,read_opd:5,read_quantunit_cont:5,read_structur:5,read_topographi:[5,11],read_varlen:5,read_with_check:5,read_x3p:5,read_xyz:5,readabl:5,reader:[4,9,11],reader_func:5,readerbas:5,readfileerror:5,readi:0,real:[4,6],realiz:4,reason:0,reciproc:8,rectangl:[6,8],recurs:5,reduc:6,refer:[2,4,5],reformat:5,reformat_dict:5,register_funct:4,relev:5,remov:[0,4,5],repeat:[4,5,8],report:4,repositori:3,repres:[4,5],represent:[4,11],request:5,requir:[3,5,8],res:5,rescal:[4,11],resolut:[4,5],respect:[4,5,6,8],restructuredtext:0,result:[0,4],retur:4,return_map:8,rewrit:0,rfn:4,rich:[2,11],right:[4,6,8],rigid:11,rmax:4,rms:[2,4,6,8,11],rms_curvatur:[6,8,11],rms_height:[4,6,8,11],rms_laplacian:8,rms_slope:[4,6,8,11],robust:5,rolloff:4,root:[0,4,6,8,11],rotation:4,rough:[4,6,8],rubber:4,run:[0,1,3,10,11],runner:0,runtest:[0,3,10],sai:0,same:[0,4,5,8],save:[4,5],save_npi:5,scalar:[6,8,11],scalarparamet:[4,9],scale:[4,5,6,11],scale_factor:4,scalednonuniformtopographi:4,scaleduniformtopographi:4,scan:[0,4,5,6,8,11],scheme:8,scipi:[5,8],script:[0,1,11],seal:4,search:[2,6],second:[5,6],section:[6,8,11],see:[0,4,5,6,8],seed:4,seen:11,segement:6,self:[4,6],self_affine_prefactor:4,seri:6,serial:[0,3,4],set:[0,1,2,3,4,6,8,11],setup:[0,1,3],sever:5,shannon:4,shape:8,shift:8,shift_and_tilt:8,short_cutoff:[4,6],short_wavelength_cutoff:8,shortcut:0,shortcut_filtered_topographi:8,shortcuttopographi:8,shortest:[4,6],should:[0,5],show:0,side:6,sign:5,signal:[4,8],similar:11,simpl:[4,8],simplest:5,simultan:5,sin:6,sinc:[4,5,6],singl:0,size:[4,5,8],size_i:5,size_x:5,skipif:0,slice:5,slightli:0,slope:[4,6,8,11],small:[4,6,7,8],smaller:8,smallest:4,soft:11,soft_wal:11,softwar:11,solut:[8,11],solv:6,solver:11,some:[2,5,11],sometim:3,sort:6,sourc:[0,1,2,4,5,6,7,8,10],space:[4,6,8,11],special:[2,5,9],specif:4,specifi:[0,1,3,4,5,8],spectral:[4,6,8],spectrum:[2,4,8,11],sphere:4,sphinx:0,split:8,sqrt:[4,8],squar:[4,6,8,11],squeez:[4,11],stai:4,standard:[0,2,4,5],standoff:4,start:[0,5,11],statement:0,statist:2,step:[5,8],stevenj:8,stochast:4,store:[4,6,8],str:[4,5,6,8],straight:[6,11],stream:5,string:[0,4,5,11],stringify_plan:4,structur:[5,11],stub:[4,8,11],style:2,subclass:5,subdirectori:11,subdivid:[6,8],subdivided_line_scan:6,subdivis:[6,8],subdomain:[5,8],subdomain_loc:[4,5,8],subdomain_slic:4,submodul:9,subpackag:9,subplot:0,subplot_loc:8,substract:[4,6,8],substrat:[0,11],subtract:[6,8],suffix:0,sum:8,sum_:[6,8],support:[4,6],sure:[0,1,4],surf:[2,4],surfac:[2,4,5,6,7,8],surface_tens:4,surfacetopographi:[1,11],symbol:[0,5],symmetr:4,syntax:0,system:[0,6,11],tabl:5,tag:0,take:[0,8],taken:[6,11],techniqu:[2,5],test:[2,5,11],test_parallel:0,testabl:4,tex:7,text:[4,5,6,8],than:[4,5,8,11],thei:[0,4,5,8,11],them:[3,6,8],theorem:4,therefor:11,thi:[0,1,2,4,5,6,8,11],thing:0,third:5,three:11,through:[4,11],tick:5,tild:[6,8],tilt:[6,8],tilt_and_curvatur:8,tilt_from_height:8,time:[4,6,8],togeth:11,tol:6,toler:[6,8],tool:[2,4,9,11],top:[5,8],topgogr:4,topgographi:8,topo:11,topogographi:4,topogr:2,topograpgi:5,topographi:[2,4,5,6,8],topography_a:4,topography_b:4,topographyinterfac:4,total:[6,8],track:5,tranpos:4,transform:[4,6,8],translat:4,translated_topographi:4,translatedtopographi:4,transposeduniformtopographi:4,treat:6,trend:[4,6,8],triangl:6,tricki:4,trivial:4,tst:0,tupl:[4,5,6,8],turn:8,twice:8,two:[4,6,8,11],txt:[3,5],type:[0,4,5,6,8],typic:[5,11],typo:0,unabl:3,undefin:[4,5],undeform:4,undeformed_profil:4,under:11,underli:4,understand:0,uniform:[2,4,6,9,11],uniformli:11,uniformlinescan:[4,6,8,11],uniformlinescanandtopographi:[8,9],uniformlyinterpolatedlinescan:4,uniformtopographi:[4,8],uniformtopographyinterfac:4,uninstal:3,uniqu:5,unit1:5,unit1_str:5,unit2:5,unit2_str:5,unit:[0,4,5,6,8,11],uniti:8,unittest:0,unknownfileformatgiven:5,unspecifi:8,updat:2,upper:[5,6,8],upper_bound:[6,8],upwind:8,usag:2,use:[0,1,2,5,6,8,11],used:[0,4,5,6,8,11],useful:[4,10,11],user:[3,6],uses:[6,11],using:[0,2,4,6,8,10,11],usr:3,util:4,valu:[0,4,5,6,8,11],vanish:4,vari:0,variabl:[2,3,5,6,8],variable_bandwidth:[6,8],variablebandwidth:[4,9],variant:11,vector:[7,8],veri:4,versa:4,version:3,via:[3,11],vice:4,voltag:5,wai:[0,6],wall:11,want:4,warn:4,wave:[4,8],wavelength:[4,8],wavevector:[6,7],well:[4,5],wether:5,what:0,when:[0,4,5,11],whenev:[0,1],where:[3,5,6,8],wherea:4,whether:[3,4,5,8],which:[0,1,5,6,8],whole:[0,8,11],window:[4,5,6,8,10],wip:0,wise:6,within:[5,6,8],without:[0,1,5,8],work:[0,11],workflow:5,would:5,wrap:4,wrapasnonuniformlinescan:4,wrappedread:5,write:[2,5],write_nc:5,written:0,x3p:5,x3preader:5,x_i:[6,8],x_rang:4,xre:5,xterm:10,xyzread:5,y_j:8,year:0,yield:6,you:[0,1,2,3,4,5,8,10,11],your:[0,3,4,10],yourself:0,yre:5,zero:[4,8],zes:5,zsensor:5},titles:["Contributing to SurfaceTopography","Development","Welcome to SurfaceTopography\u2019s documentation!","Installation","SurfaceTopography package","SurfaceTopography.IO package","SurfaceTopography.Nonuniform package","SurfaceTopography.Tools package","SurfaceTopography.Uniform package","SurfaceTopography","Testing","Usage"],titleterms:{"function":11,analysi:11,author:0,autocorrel:[6,8],branch:0,code:0,commit:0,common:[4,6,7,8],compil:0,content:[4,5,6,7,8],contribut:0,convert:4,data:11,debug:0,detrend:[6,8],develop:[0,1],direct:3,directori:3,document:[0,2],ffttrick:4,filter:8,from:3,fromfil:5,gener:4,handl:11,heightcontain:4,ibw:5,indic:2,instal:3,interpol:[4,8],lapack:3,matlab:5,modul:[4,5,6,7,8],mpi:0,nonuniform:6,nonuniformlinescan:4,npy:5,opdx:5,openbla:3,orient:11,packag:[4,5,6,7,8],pip:3,pipelin:11,plot:0,powerspectrum:[6,8],problem:3,reader:5,scalarparamet:[6,8],sourc:3,special:4,style:0,submodul:[4,5,6,7,8],subpackag:4,surfacetopographi:[0,2,3,4,5,6,7,8,9],tabl:2,test:[0,10],tool:7,topographi:11,uniform:8,uniformlinescanandtopographi:4,updat:3,usag:11,variablebandwidth:[6,8],welcom:2,write:0}})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
# plt.ion()


# # Get the current script's directory (where the data files are assumed to be)
# current_dir = os.getcwd()

# # Filenames must match exactly the original names you provided
# file_2019 = os.path.join(current_dir, "2019data6.csv")
# file_2022 = os.path.join(current_dir, "2022data6.csv")

# # Load the datasets
# data_2019 = pd.read_csv(file_2019, parse_dates=['Date'])
# data_2022 = pd.read_csv(file_2022, parse_dates=['Date and time'])



## QUESTION 3.1 B
# Define scrfft function
def scrfft(xdata, ydata):
    """
    Compute a uniformly sampled interpolation of input data and return its real Fourier coefficients.

    This function:
    1. Sorts the input data by x-values.
    2. Creates an evenly spaced x‑grid spanning the same domain as the input.
    3. Interpolates y-values onto this uniform grid.
    4. Computes the real-valued Fast Fourier Transform (rFFT) of the interpolated signal.
    5. Returns the frequency components and the corresponding cosine (a) and sine (b) coefficients.

    Parameters
    ----------
    xdata : array_like
        Input x-values. These may be unevenly spaced and are sorted internally.
    ydata : array_like
        Input y-values corresponding to each element of `xdata`.

    Returns
    -------
    f : ndarray
        Array of frequency values associated with the Fourier coefficients.
    a : ndarray
        Cosine (real) Fourier coefficients of the signal.
    b : ndarray
        Sine (negative imaginary) Fourier coefficients of the signal.

    Notes
    -----
    - The input data is interpolated to a uniform grid because FFT requires uniform sampling.
    - The DC (zero-frequency) term is halved to follow the standard Fourier series normalization.
    - The function returns only positive frequencies due to the use of rFFT.

    """
    sdata = np.argsort(xdata)
    xdatas = xdata[sdata]
    ydatas = ydata[sdata]

    xmin = np.min(xdata)
    xmax = np.max(xdata)
    ndata = len(xdata)
    x = (xmax - xmin) / (ndata - 1) * np.arange(ndata) + xmin
    y = np.interp(x, xdatas, ydatas)

    yf = 2.0 * np.fft.rfft(y) / (ndata + 1)
    a = np.real(yf)
    b = -np.imag(yf)
    a[0] = 0.5 * a[0]
    f = np.arange(len(a)) / (xmax - xmin)

    return f, a, b

# -------------------------------
# Define Fourier smoothing
# -------------------------------
def fourier_smooth(x, y, terms=8):
    """
    Smooth a signal using a truncated Fourier series reconstruction.

    This function performs Fourier smoothing by:
    1. Computing the real Fourier series coefficients of the input data via `scrfft`.
    2. Reconstructing the signal using the specified number of low-frequency terms (default: 8).
    3. Returning a smoothed version of the original signal.

    Parameters
    ----------
    x : array_like
        Input x-values (not necessarily uniformly spaced).
    y : array_like
        Input y-values corresponding to each x-value.
    terms : int, optional
        Number of Fourier terms to use in the smoothing. Default is 8.

    Returns
    -------
    x_smooth : ndarray
        Evenly spaced x-values used for the smoothed signal.
    y_smooth : ndarray
        Smoothed y-values reconstructed from the truncated Fourier series.

    Notes
    -----
    - This function depends on `scrfft` to compute Fourier coefficients.
    - Higher `terms` will include more frequency components and produce a smoother fit that follows the original data more closely.
    - Too many terms may reintroduce noise, while too few may oversimplify the signal.
    """
    f, a, b = scrfft(x, y)
    xmin, xmax = np.min(x), np.max(x)
    ndata = len(x)
    x_smooth = np.linspace(xmin, xmax, ndata)
    y_smooth = np.zeros_like(x_smooth)
    for i in range(terms):
        y_smooth += a[i] * np.cos(2 * np.pi * f[i] * x_smooth) + b[i] * np.sin(2 * np.pi * f[i] * x_smooth)
    return x_smooth, y_smooth

def plot_fourier_smoothed_passengers(data_2019, data_2022):
    """
    Generate Figure 1: Compare 2019 and 2022 public transport passenger counts over the year
    using Fourier smoothing.

    This function:
    - Processes daily total passengers for 2019 from bus and metro data.
    - Estimates daily passenger counts for 2022 using trip proportions and known yearly totals.
    - Applies Fourier smoothing to both datasets.
    - Plots scatter data and smoothed curves for both years.

    Parameters
    ----------
    data_2019 : pd.DataFrame
        Dataset with 2019 public transport data.
    data_2022 : pd.DataFrame
        Dataset with 2022 public transport data, including 'Mode' column.

    Returns
    -------
    None
        Displays the plot comparing smoothed and raw daily passenger counts for 2019 and 2022.
    """
    # Process 2019 data
    data_2019["Total Passengers"] = (
        data_2019["Bus pax number peak"] +
        data_2019["Bus pax number offpeak"] +
        data_2019["Metro pax number peak"] +
        data_2019["Metro pax number offpeak"]
    )
    data_2019["Date"] = pd.to_datetime(data_2019["Date"])
    data_2019["Day of Year"] = data_2019["Date"].dt.dayofyear
    data_2019_sorted = data_2019.sort_values("Day of Year")
    x_2019 = data_2019_sorted["Day of Year"].values
    y_2019 = data_2019_sorted["Total Passengers"].values

    # Process 2022 data
    data_2022["Date and time"] = pd.to_datetime(data_2022["Date and time"])
    data_2022["Day of Year"] = data_2022["Date and time"].dt.dayofyear

    daily_trips = data_2022.groupby(["Day of Year", "Mode"]).size().unstack(fill_value=0)
    total_trips_year = daily_trips.sum()
    total_passengers_year = {
        'Bus': 39537756,
        'Tram': 0,
        'Metro': 7064380
    }

    for mode in ['Bus', 'Tram', 'Metro']:
        if mode in daily_trips.columns:
            daily_trips[mode + " Passengers"] = (
                daily_trips[mode] / total_trips_year[mode] * total_passengers_year[mode]
            )
        else:
            daily_trips[mode + " Passengers"] = 0

    daily_trips['Total Passengers'] = daily_trips[['Bus Passengers', 'Tram Passengers', 'Metro Passengers']].sum(axis=1)
    daily_trips = daily_trips.reset_index()
    daily_trips_sorted = daily_trips.sort_values("Day of Year")

    x_2022 = daily_trips_sorted["Day of Year"].values
    y_2022 = daily_trips_sorted["Total Passengers"].values

    # Apply Fourier smoothing
    x2019_smooth, y2019_smooth = fourier_smooth(x_2019, y_2019)
    x2022_smooth, y2022_smooth = fourier_smooth(x_2022, y_2022)

    # Smoothed x values
    x_smooth_days_2019 = np.linspace(x_2019.min(), x_2019.max(), len(y2019_smooth))
    x_smooth_days_2022 = np.linspace(x_2022.min(), x_2022.max(), len(y2022_smooth))

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(x_2019, y_2019, s=10, label="2019 Scatter", alpha=0.6, color='blue')
    ax.plot(x_smooth_days_2019, y2019_smooth, label="2019 Smoothed (Fourier)", linewidth=2, color='blue')

    ax.scatter(x_2022, y_2022, s=10, label="2022 Scatter", alpha=0.6, color='orange')
    ax.plot(x_smooth_days_2022, y2022_smooth, label="2022 Smoothed (Fourier)", linewidth=2, color='orange')

    ax.set_title("Figure 1: Daily Public Transport Passengers with Fourier Smoothed Lines (Raw Values)\nStudent ID: 24157966", fontsize=16)
    ax.set_xlabel("Day of the Year")
    ax.set_ylabel("Passenger Count")
    ax.set_xlim(1, 366)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------
## QUESTION 3.1 C
# -----------------------------------------------------------------------------------------------------------------------------------
def plot_bar_avgpax_weekdays(data_2019, data_2022, 
                                          X_spring_2019, Y_summer_2019, Z_autumn_2019,
                                          X_spring_2022, Y_summer_2022, Z_autumn_2022):
    """
    Generate a bar plot comparing the average number of public transport passengers per weekday 
    for the years 2019 and 2022, along with seasonal journey percentages.

    This function:
    - Processes raw 2019 and 2022 transport datasets.
    - Calculates the average number of passengers per weekday for both years.
    - Uses actual passenger counts for 2019.
    - Estimates 2022 passenger counts using proportional scaling from known total yearly values.
    - Displays a side-by-side bar plot for weekday averages.
    - Annotates the plot with spring/summer/autumn percentages for both years and student ID.

    Parameters
    ----------
    data_2019 : pd.DataFrame
        DataFrame containing 2019 public transport data, with date and passenger columns for Bus and Metro.
    data_2022 : pd.DataFrame
        DataFrame containing 2022 public transport data, with datetime and mode columns.
    X_spring_2019 : float
        Percentage of journeys made in spring (March–May) in 2019.
    Y_summer_2019 : float
        Percentage of journeys made in summer (June–August) in 2019.
    Z_autumn_2019 : float
        Percentage of journeys made in autumn (September–November) in 2019.
    X_spring_2022 : float
        Percentage of journeys made in spring (March–May) in 2022.
    Y_summer_2022 : float
        Percentage of journeys made in summer (June–August) in 2022.
    Z_autumn_2022 : float
        Percentage of journeys made in autumn (September–November) in 2022.

    Returns
    -------
    None
        Displays a single-panel bar chart comparing weekday passenger averages between 2019 and 2022.
    
    Notes
    -----
    - Passenger numbers for 2022 are scaled estimates based on the number of mode-specific trips
      and known total yearly passengers for Bus, Tram, and Metro.
    - The figure includes a boxed annotation showing seasonal percentages and student ID.
    - This figure is intended for use as Figure 2 in a transport analysis report.
    """
    # Convert dates
    data_2019['Date'] = pd.to_datetime(data_2019['Date'])
    data_2022['Date and time'] = pd.to_datetime(data_2022['Date and time'])

    # ---- 2019: Total passengers already available ----
    data_2019['Total Passengers'] = (
        data_2019['Bus pax number peak'] + data_2019['Bus pax number offpeak'] +
        data_2019['Metro pax number peak'] + data_2019['Metro pax number offpeak']
    )
    data_2019['DayOfWeek_2019'] = data_2019['Date'].dt.day_name()

    avg_passengers_2019 = data_2019.groupby('DayOfWeek_2019')['Total Passengers'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])

    # ---- 2022: Estimate passengers using total yearly passengers ----
    total_passengers_year = {
        'Bus': 39537756,
        'Tram': 0,
        'Metro': 7064380
    }

    data_2022['Day of Year'] = data_2022['Date and time'].dt.dayofyear
    data_2022['Weekday'] = data_2022['Date and time'].dt.day_name()

    daily_trips = data_2022.groupby(['Day of Year', 'Mode']).size().unstack(fill_value=0)
    total_trips_year = daily_trips.sum()

    for mode in ['Bus', 'Tram', 'Metro']:
        if mode in daily_trips.columns:
            daily_trips[mode + ' Passengers'] = (
                daily_trips[mode] / total_trips_year[mode] * total_passengers_year[mode]
            )
        else:
            daily_trips[mode + ' Passengers'] = 0

    daily_trips['Total Passengers'] = daily_trips[['Bus Passengers', 'Tram Passengers', 'Metro Passengers']].sum(axis=1)

    daily_trips = daily_trips.reset_index()
    weekday_lookup = data_2022.drop_duplicates('Day of Year')[['Day of Year', 'Weekday']]
    daily_trips = pd.merge(daily_trips, weekday_lookup, on='Day of Year', how='left')

    avg_passengers_2022 = daily_trips.groupby('Weekday')['Total Passengers'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])

    # -------------------------------
    # Single-panel bar plot
    # -------------------------------
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    x = range(len(weekdays))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(14, 7))
    bars_2019 = ax.bar([i - bar_width/2 for i in x], avg_passengers_2019.values, width=bar_width,
                       label='2019', color='skyblue')
    bars_2022 = ax.bar([i + bar_width/2 for i in x], avg_passengers_2022.values, width=bar_width,
                       label='2022', color='lightgreen')

    # Add text annotations
    for bar in bars_2019:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2000, f'{height:,.0f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars_2022:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2000, f'{height:,.0f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_title('Figure 2: Average Public Transport Passengers by Day (2019 vs 2022)', fontsize=15, pad=45)

    seasonal_text = (
        f"2019 - Spring: {X_spring_2019:.2f}%, Summer: {Y_summer_2019:.2f}%, Autumn: {Z_autumn_2019:.2f}%    |    "
        f"2022 - Spring: {X_spring_2022:.2f}%, Summer: {Y_summer_2022:.2f}%, Autumn: {Z_autumn_2022:.2f}%"
    )
    ax.annotate(seasonal_text, xy=(0.5, 1.02), xycoords='axes fraction', fontsize=11,
                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))
    
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Average Number of Passengers')
    ax.set_xticks(x)
    ax.set_xticklabels(weekdays, rotation=45)
    ax.set_ylim(0, max(max(avg_passengers_2019.max(), avg_passengers_2022.max()) * 1.15, 1))
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.text(0.5, 0.01, 'Student ID: 24157966', ha='center', fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
# -----------------------------------------------------------------------------------------------------------------------------------
## QUESTION 3.1 D
# -----------------------------------------------------------------------------------------------------------------------------------
def plot_metro_linear_fit(df):
    """
    Creates a scatter plot of Price vs Distance for all Metro trips,
    fits a linear regression, draws the regression line,
    and displays the linear equation on the figure.
    """

    # Filter Metro journeys
    metro = df[df["Mode"].str.lower() == "metro"]

    # Prepare X and y
    X = metro[["Distance"]].values
    y = metro["Price"].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Generate smooth line for plotting
    xx = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    yy = model.predict(xx)

    # Extract coefficients
    slope = model.coef_[0]
    intercept = model.intercept_

    # Create equation string
    equation = f"Price = {slope:.3f} × Distance + {intercept:.3f}"

    # Plot
    plt.figure(figsize=(9, 6))
    plt.scatter(X, y, alpha=0.6, label="Metro journeys (2022)", color="orange")
    plt.plot(xx, yy, color="red", linewidth=2, label="Linear regression fit")
    
    student_id = "24157966"
    plt.text(4.5, 0, f'Student ID: {student_id}', ha='center', fontsize=10)
    
    # Add labels and title
    plt.xlabel("Trip Length (km)")
    plt.ylabel("Price (Euros)")
    plt.title("Figure 3: Metro Journeys 2022: Price vs Trip Length with Linear Fit")
    plt.grid(True)
    plt.legend()
    

    # Display equation on plot
    plt.text(
        0.05, 0.95,
        equation,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    plt.show()

    return model
# -----------------------------------------------------------------------------------------------------------------------------------
## QUESTION 3.2
# -----------------------------------------------------------------------------------------------------------------------------------
def transport_type_seasonwise_plot(data_2019, data_2022):
    """
    Generate a bar plot comparing the percentage of journeys by transport mode in 2019 and 2022,
    and calculate the seasonal distribution of journeys (spring, summer, autumn) for both years.

    This function:
    - Computes the share of journeys made by Bus, Tram, Train, and Metro in both 2019 and 2022.
    - For 2019: Calculates total passenger numbers from peak/off-peak values.
    - For 2022: Uses mode counts to estimate journey share.
    - Calculates the percentage of total journeys made during spring, summer, and autumn months.
    - Displays a bar chart showing mode share percentages for each year.

    Parameters
    ----------
    data_2019 : pd.DataFrame
        Public transport dataset for 2019. Must include columns:
        - 'Date', 'Bus pax number peak', 'Bus pax number offpeak',
          'Metro pax number peak', 'Metro pax number offpeak'.
    data_2022 : pd.DataFrame
        Public transport dataset for 2022. Must include columns:
        - 'Date and time', 'Mode' (with values like 'Bus', 'Metro', etc.).

    Returns
    -------
    dict
        Dictionary containing the percentage of total journeys made in each season:
        {
            "Spring (%) for 2019": float,
            "Summer (%) for 2019": float,
            "Autumn (%) for 2019": float,
            "Spring (%) for 2022": float,
            "Summer (%) for 2022": float,
            "Autumn (%) for 2022": float
        }

    Notes
    -----
    - The bar chart shows transport mode share for 2019 and 2022.
    - Seasonal journey distribution is based on calendar months:
        * Spring: March–May
        * Summer: June–August
        * Autumn: September–November
    - This figure is labeled as "Figure 4" and includes the student ID in the chart.

    """
    # Process 2019 data
    data_2019['Date'] = pd.to_datetime(data_2019['Date'])
    data_2019['Total_Bus'] = data_2019['Bus pax number peak'] + data_2019['Bus pax number offpeak']
    data_2019['Total_Metro'] = data_2019['Metro pax number peak'] + data_2019['Metro pax number offpeak']
    total_2019_bus = data_2019['Total_Bus'].sum()
    total_2019_metro = data_2019['Total_Metro'].sum()
    total_2019_all = total_2019_bus + total_2019_metro

    percent_2019 = {
        'Tram': 0.0,
        'Train': 0.0,
        'Bus': (total_2019_bus / total_2019_all) * 100,
        'Metro': (total_2019_metro / total_2019_all) * 100
    }

    # Process 2022 data
    data_2022['Date and time'] = pd.to_datetime(data_2022['Date and time'])
    mode_counts = data_2022['Mode'].value_counts()
    total_2022_all = mode_counts.sum()
    percent_2022 = {
        'Tram': (mode_counts.get('Tram', 0) / total_2022_all) * 100,
        'Train': (mode_counts.get('Train', 0) / total_2022_all) * 100,
        'Bus': (mode_counts.get('Bus', 0) / total_2022_all) * 100,
        'Metro': (mode_counts.get('Metro', 0) / total_2022_all) * 100
    }

    # Create transport percentage DataFrame
    transport_df = pd.DataFrame({'2019': percent_2019, '2022': percent_2022})

    # Seasonal breakdown 2019
    spring_months, summer_months, autumn_months = [3, 4, 5], [6, 7, 8], [9, 10, 11]
    data_2019['Month'] = data_2019['Date'].dt.month
    spring_total = data_2019[data_2019['Month'].isin(spring_months)][['Total_Bus', 'Total_Metro']].sum().sum()
    summer_total = data_2019[data_2019['Month'].isin(summer_months)][['Total_Bus', 'Total_Metro']].sum().sum()
    autumn_total = data_2019[data_2019['Month'].isin(autumn_months)][['Total_Bus', 'Total_Metro']].sum().sum()
    total_journeys_2019 = total_2019_all
    X_spring_2019 = (spring_total / total_journeys_2019) * 100
    Y_summer_2019 = (summer_total / total_journeys_2019) * 100
    Z_autumn_2019 = (autumn_total / total_journeys_2019) * 100

    # Seasonal breakdown 2022
    data_2022['Month'] = data_2022['Date and time'].dt.month
    spring_total_2022 = len(data_2022[data_2022['Month'].isin(spring_months)])
    summer_total_2022 = len(data_2022[data_2022['Month'].isin(summer_months)])
    autumn_total_2022 = len(data_2022[data_2022['Month'].isin(autumn_months)])
    total_journeys_2022 = len(data_2022)
    X_spring_2022 = (spring_total_2022 / total_journeys_2022) * 100
    Y_summer_2022 = (summer_total_2022 / total_journeys_2022) * 100
    Z_autumn_2022 = (autumn_total_2022 / total_journeys_2022) * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    modes = transport_df.index
    x = range(len(modes))
    bar_width = 0.35
    bars_2019 = ax.bar([i - bar_width / 2 for i in x], transport_df['2019'], width=bar_width, label='2019')
    bars_2022 = ax.bar([i + bar_width / 2 for i in x], transport_df['2022'], width=bar_width, label='2022')

    for bar in bars_2019 + bars_2022:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%', ha='center', va='bottom')

    ax.set_title('Figure 4: Percentage of Journeys by Mode (2019 vs 2022)')
    ax.set_xlabel('Mode of Transport')
    ax.set_ylabel('Percentage of Journeys (%)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 100)
    ax.legend()
    plt.text(1.5, -15, 'Student ID: 24157966', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()

    return {
        "Spring (%) for 2019": round(X_spring_2019, 2),
        "Summer (%) for 2019": round(Y_summer_2019, 2),
        "Autumn (%) for 2019": round(Z_autumn_2019, 2),
        "Spring (%) for 2022": round(X_spring_2022, 2),
        "Summer (%) for 2022": round(Y_summer_2022, 2),
        "Autumn (%) for 2022": round(Z_autumn_2022, 2)
    }

# seasonal_percentages = transport_type_seasonwise_plot(data_2019, data_2022)
# plot_metro_linear_fit(data_2022)
# plot_bar_avgpax_weekdays(
#     data_2019,
#     data_2022,
#     X_spring_2019=seasonal_percentages["Spring (%) for 2019"],
#     Y_summer_2019=seasonal_percentages["Summer (%) for 2019"],
#     Z_autumn_2019=seasonal_percentages["Autumn (%) for 2019"],
#     X_spring_2022=seasonal_percentages["Spring (%) for 2022"],
#     Y_summer_2022=seasonal_percentages["Summer (%) for 2022"],
#     Z_autumn_2022=seasonal_percentages["Autumn (%) for 2022"]
# )

def main():
    # Get the current script's directory (where the data files are assumed to be)
    current_dir = os.getcwd()

    # Filenames must match exactly the original names you provided
    file_2019 = os.path.join(current_dir, "2019data6.csv")
    file_2022 = os.path.join(current_dir, "2022data6.csv")

    # Load the datasets
    data_2019 = pd.read_csv(file_2019, parse_dates=['Date'])
    data_2022 = pd.read_csv(file_2022, parse_dates=['Date and time'])

    # Figure 1: Fourier smoothing plot
    plot_fourier_smoothed_passengers(data_2019, data_2022)

    # Figure 4: Transport mode comparison and seasonal breakdown
    seasonal_percentages = transport_type_seasonwise_plot(data_2019, data_2022)

    # Figure 3: Metro trend line (assuming you have this function already)
    plot_metro_linear_fit(data_2022)

    # Figure 2: Weekday averages + seasonal annotation
    plot_bar_avgpax_weekdays(
        data_2019,
        data_2022,
        X_spring_2019=seasonal_percentages["Spring (%) for 2019"],
        Y_summer_2019=seasonal_percentages["Summer (%) for 2019"],
        Z_autumn_2019=seasonal_percentages["Autumn (%) for 2019"],
        X_spring_2022=seasonal_percentages["Spring (%) for 2022"],
        Y_summer_2022=seasonal_percentages["Summer (%) for 2022"],
        Z_autumn_2022=seasonal_percentages["Autumn (%) for 2022"]
    )

# Standard Python script entry point
if __name__ == "__main__":
    main()




